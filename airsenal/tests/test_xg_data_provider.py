"""
Comprehensive tests for xG/xA data provider integration.

Tests cover:
- Provider interface and implementations
- Circuit breaker pattern
- Rate limiting and retry logic
- Data transformation
- API health monitoring
- Error handling and graceful degradation
"""

import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

from airsenal.framework.xg_data_provider import (
    XGDataProvider,
    SportmonksProvider,
    XGDataManager,
    APIHealthMonitor,
    DataTransformer,
    CircuitBreaker,
    CircuitBreakerState,
    ProviderConfig,
    ProviderTier,
    XGDataPoint,
    APIUsageMetrics,
    create_xg_manager
)


class TestProviderConfig:
    """Test provider configuration validation"""
    
    def test_valid_config(self):
        """Test creating valid configuration"""
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://api.example.com",
            rate_limit_per_minute=100,
            timeout_seconds=30
        )
        assert config.api_key == "test_key"
        assert config.rate_limit_per_minute == 100
        assert config.timeout_seconds == 30
    
    def test_invalid_rate_limit(self):
        """Test validation of rate limit"""
        with pytest.raises(ValueError, match="Rate limit must be positive"):
            ProviderConfig(rate_limit_per_minute=0)
    
    def test_invalid_timeout(self):
        """Test validation of timeout"""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            ProviderConfig(timeout_seconds=0)


class TestXGDataPoint:
    """Test xG data point structure and calculations"""
    
    def test_basic_xg_data_point(self):
        """Test creating basic xG data point"""
        xg_point = XGDataPoint(
            player_id=123,
            fixture_id=456,
            match_date=datetime.now(),
            expected_goals=0.5,
            expected_assists=0.3
        )
        
        assert xg_point.player_id == 123
        assert xg_point.fixture_id == 456
        assert xg_point.expected_goals == 0.5
        assert xg_point.expected_assists == 0.3
        # Should auto-calculate xGI
        assert xg_point.expected_goal_involvements == 0.8
    
    def test_xg_data_point_with_xgi(self):
        """Test xG data point with explicit xGI (no auto-calculation)"""
        xg_point = XGDataPoint(
            player_id=123,
            fixture_id=456,
            match_date=datetime.now(),
            expected_goals=0.5,
            expected_assists=0.3,
            expected_goal_involvements=0.9  # Explicit value
        )
        
        # Should keep explicit value, not auto-calculate
        assert xg_point.expected_goal_involvements == 0.9


class TestAPIUsageMetrics:
    """Test API usage tracking and metrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = APIUsageMetrics("TestProvider")
        assert metrics.provider_name == "TestProvider"
        assert metrics.requests_made == 0
        assert metrics.success_rate == 1.0
    
    def test_add_successful_request(self):
        """Test recording successful request"""
        metrics = APIUsageMetrics("TestProvider")
        metrics.add_request(success=True, response_time=0.5, cost=0.01)
        
        assert metrics.requests_made == 1
        assert metrics.requests_failed == 0
        assert metrics.success_rate == 1.0
        assert metrics.total_cost == 0.01
        assert metrics.average_response_time == 0.5
    
    def test_add_failed_request(self):
        """Test recording failed request"""
        metrics = APIUsageMetrics("TestProvider")
        metrics.add_request(success=False, response_time=1.0)
        
        assert metrics.requests_made == 1
        assert metrics.requests_failed == 1
        assert metrics.success_rate == 0.0
        assert metrics.total_cost == 0.0
    
    def test_mixed_requests_success_rate(self):
        """Test success rate calculation with mixed results"""
        metrics = APIUsageMetrics("TestProvider")
        
        # 3 successful, 2 failed = 60% success rate
        for _ in range(3):
            metrics.add_request(success=True, response_time=0.5)
        for _ in range(2):
            metrics.add_request(success=False, response_time=1.0)
        
        assert metrics.requests_made == 5
        assert metrics.requests_failed == 2
        assert metrics.success_rate == 0.6
    
    def test_rate_limit_window(self):
        """Test rate limiting window calculations"""
        metrics = APIUsageMetrics("TestProvider", daily_limit=100)
        
        # Add requests within window
        for _ in range(5):
            metrics.add_request(success=True, response_time=0.5)
        
        assert metrics.current_window_requests == 5
        assert metrics.requests_remaining_in_window == 95


class TestCircuitBreaker:
    """Test circuit breaker implementation"""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def failing_func():
            raise Exception("API Error")
        
        # Trigger failures up to threshold
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        # Circuit breaker should now be open
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.failure_count == 3
    
    def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        
        def failing_func():
            raise Exception("API Error")
        
        # Trigger failures to open circuit
        for i in range(2):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        # Should reject new calls
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            breaker.call(lambda: "test")
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery mechanism"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        def failing_func():
            raise Exception("API Error")
        
        def success_func():
            return "success"
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should allow one test call (half-open)
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED


class TestDataTransformer:
    """Test data transformation from provider formats"""
    
    def test_sportmonks_date_parsing(self):
        """Test parsing various Sportmonks date formats"""
        transformer = DataTransformer()
        
        # ISO format with Z
        date1 = transformer._parse_sportmonks_date("2023-12-25T15:30:00Z")
        assert isinstance(date1, datetime)
        
        # ISO format without Z
        date2 = transformer._parse_sportmonks_date("2023-12-25T15:30:00")
        assert isinstance(date2, datetime)
        
        # Invalid date should return current time
        date3 = transformer._parse_sportmonks_date("invalid")
        assert isinstance(date3, datetime)
        
        # None should return current time
        date4 = transformer._parse_sportmonks_date(None)
        assert isinstance(date4, datetime)
    
    def test_extract_xg_stats_from_sportmonks(self):
        """Test extracting xG statistics from Sportmonks data"""
        transformer = DataTransformer()
        
        stat_data = {
            "data": {
                "expected_goals": 0.5,
                "expected_assists": 0.3,
                "xgi": 0.8
            }
        }
        
        xg_stats = transformer._extract_xg_stats_from_sportmonks(stat_data)
        
        assert xg_stats["expected_goals"] == 0.5
        assert xg_stats["expected_assists"] == 0.3
        assert xg_stats["expected_goal_involvements"] == 0.8
    
    def test_sportmonks_to_xg_data_transformation(self):
        """Test full Sportmonks response transformation"""
        transformer = DataTransformer()
        
        response_data = {
            "data": [
                {
                    "fixture_id": 123,
                    "date": "2023-12-25T15:30:00Z",
                    "data": {
                        "expected_goals": 0.5,
                        "expected_assists": 0.3
                    }
                }
            ]
        }
        
        xg_data = transformer.sportmonks_to_xg_data(response_data, player_id=456)
        
        assert len(xg_data) == 1
        assert xg_data[0].player_id == 456
        assert xg_data[0].fixture_id == 123
        assert xg_data[0].expected_goals == 0.5
        assert xg_data[0].expected_assists == 0.3
        assert xg_data[0].provider == "Sportmonks"


class MockProvider(XGDataProvider):
    """Mock provider for testing abstract base class"""
    
    def get_player_xg_data(self, player_id, fixture_id=None, date_range=None):
        return []
    
    def get_match_xg_data(self, fixture_id):
        return []
    
    def get_bulk_xg_data(self, date_range, gameweeks=None):
        return []
    
    def _get_auth_headers(self):
        return {"Authorization": "Bearer test"}
    
    def _is_credentials_optional(self):
        return True


class TestXGDataProvider:
    """Test abstract base provider functionality"""
    
    def test_provider_initialization(self):
        """Test provider initialization"""
        config = ProviderConfig(api_key="test_key")
        provider = MockProvider(config)
        
        assert provider.config.api_key == "test_key"
        assert isinstance(provider.metrics, APIUsageMetrics)
        assert isinstance(provider.circuit_breaker, CircuitBreaker)
    
    @patch('requests.Session.get')
    def test_successful_api_request(self, mock_get):
        """Test successful API request handling"""
        config = ProviderConfig(api_key="test_key")
        provider = MockProvider(config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = provider._make_request("https://api.test.com/endpoint")
        
        assert result == {"data": "test"}
        assert provider.metrics.requests_made == 1
        assert provider.metrics.requests_failed == 0
    
    @patch('requests.Session.get')
    def test_failed_api_request(self, mock_get):
        """Test failed API request handling"""
        config = ProviderConfig(api_key="test_key")
        provider = MockProvider(config)
        
        # Mock failed response
        mock_get.side_effect = HTTPError("API Error")
        
        with pytest.raises(HTTPError):
            provider._make_request("https://api.test.com/endpoint")
        
        assert provider.metrics.requests_made == 1
        assert provider.metrics.requests_failed == 1
    
    def test_rate_limit_check(self):
        """Test rate limiting functionality"""
        config = ProviderConfig(rate_limit_per_minute=2)
        provider = MockProvider(config)
        
        # Should allow first 2 requests
        assert provider._check_rate_limit() == True
        provider.metrics.current_window_requests = 1
        assert provider._check_rate_limit() == True
        
        # Should block 3rd request
        provider.metrics.current_window_requests = 2
        assert provider._check_rate_limit() == False


class TestSportmonksProvider:
    """Test Sportmonks provider implementation"""
    
    def test_sportmonks_initialization_with_config(self):
        """Test Sportmonks provider initialization with config"""
        config = ProviderConfig(
            api_key="test_key",
            base_url="https://api.sportmonks.com/v3/football/"
        )
        provider = SportmonksProvider(config)
        
        assert provider.config.api_key == "test_key"
        assert "sportmonks.com" in provider.config.base_url
    
    def test_sportmonks_initialization_without_config(self):
        """Test Sportmonks provider initialization without config"""
        with patch('airsenal.framework.xg_data_provider.SPORTMONKS_API_KEY', None):
            provider = SportmonksProvider()
            
            assert provider.config.api_key is None
            assert "sportmonks.com" in provider.config.base_url
    
    def test_sportmonks_auth_headers(self):
        """Test Sportmonks authentication headers"""
        config = ProviderConfig(api_key="test_key")
        provider = SportmonksProvider(config)
        
        headers = provider._get_auth_headers()
        assert headers["Authorization"] == "Bearer test_key"
    
    def test_sportmonks_auth_headers_no_key(self):
        """Test Sportmonks headers without API key"""
        config = ProviderConfig(api_key=None)
        provider = SportmonksProvider(config)
        
        headers = provider._get_auth_headers()
        assert headers == {}
    
    @patch.object(SportmonksProvider, '_make_request')
    def test_get_player_xg_data(self, mock_request):
        """Test fetching player xG data"""
        provider = SportmonksProvider()
        
        # Mock API response
        mock_response = {
            "data": [
                {
                    "fixture_id": 123,
                    "date": "2023-12-25T15:30:00Z",
                    "data": {
                        "expected_goals": 0.5,
                        "expected_assists": 0.3
                    }
                }
            ]
        }
        mock_request.return_value = mock_response
        
        result = provider.get_player_xg_data(player_id=456)
        
        assert len(result) == 1
        assert result[0].player_id == 456
        assert result[0].fixture_id == 123
        assert result[0].expected_goals == 0.5
    
    @patch.object(SportmonksProvider, '_make_request')
    def test_get_match_xg_data(self, mock_request):
        """Test fetching match xG data"""
        provider = SportmonksProvider()
        
        # Mock API response
        mock_response = {
            "data": [
                {
                    "player": {"id": 456},
                    "date": "2023-12-25T15:30:00Z",
                    "data": {
                        "expected_goals": 0.5,
                        "expected_assists": 0.3
                    }
                }
            ]
        }
        mock_request.return_value = mock_response
        
        result = provider.get_match_xg_data(fixture_id=123)
        
        assert len(result) == 1
        assert result[0].player_id == 456
        assert result[0].fixture_id == 123
    
    @patch.object(SportmonksProvider, '_make_request')
    def test_get_bulk_xg_data(self, mock_request):
        """Test fetching bulk xG data"""
        provider = SportmonksProvider()
        
        # Mock API response
        mock_response = {
            "data": [
                {
                    "id": 123,
                    "starting_at": "2023-12-25T15:30:00Z",
                    "round": {"name": "Regular Season - 15"},
                    "statistics": [
                        {
                            "player": {"id": 456},
                            "data": {
                                "expected_goals": 0.5,
                                "expected_assists": 0.3
                            }
                        }
                    ]
                }
            ],
            "meta": {"has_more_pages": False}
        }
        mock_request.return_value = mock_response
        
        date_range = (datetime.now() - timedelta(days=7), datetime.now())
        result = provider.get_bulk_xg_data(date_range)
        
        assert len(result) == 1
        assert result[0].player_id == 456
        assert result[0].fixture_id == 123


class TestAPIHealthMonitor:
    """Test API health monitoring"""
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = APIHealthMonitor()
        assert len(monitor.providers) == 0
    
    def test_register_provider(self):
        """Test registering provider for monitoring"""
        monitor = APIHealthMonitor()
        config = ProviderConfig()
        provider = MockProvider(config)
        
        monitor.register_provider("test", provider)
        assert "test" in monitor.providers
    
    def test_get_health_status(self):
        """Test getting health status"""
        monitor = APIHealthMonitor()
        config = ProviderConfig()
        provider = MockProvider(config)
        
        # Add some metrics
        provider.metrics.add_request(True, 0.5, 0.01)
        provider.metrics.add_request(False, 1.0)
        
        monitor.register_provider("test", provider)
        status = monitor.get_health_status()
        
        assert "providers" in status
        assert "test" in status["providers"]
        assert status["providers"]["test"]["total_requests"] == 2
        assert status["providers"]["test"]["failed_requests"] == 1
        assert status["providers"]["test"]["success_rate"] == 0.5
    
    def test_cost_summary(self):
        """Test cost summary generation"""
        monitor = APIHealthMonitor()
        config = ProviderConfig(cost_per_request=0.01)
        provider = MockProvider(config)
        
        # Add some requests with costs
        provider.metrics.add_request(True, 0.5, 0.01)
        provider.metrics.add_request(True, 0.5, 0.01)
        
        monitor.register_provider("test", provider)
        cost_summary = monitor.get_cost_summary()
        
        assert cost_summary["total_cost"] == 0.02
        assert cost_summary["total_requests"] == 2
        assert cost_summary["average_cost_per_request"] == 0.01
    
    def test_recommendations_generation(self):
        """Test generating recommendations"""
        monitor = APIHealthMonitor()
        config = ProviderConfig()
        provider = MockProvider(config)
        
        # Add metrics that trigger recommendations
        for _ in range(20):  # Many requests for reliability calculation
            provider.metrics.add_request(False, 0.5)  # All failures
        
        provider.metrics.rate_limit_hits = 10  # Many rate limit hits
        provider.metrics.average_response_time = 15  # Slow responses
        
        monitor.register_provider("test", provider)
        recommendations = monitor._get_recommendations()
        
        # Should have recommendations for all issues
        recommendation_types = [r["type"] for r in recommendations]
        assert "reliability" in recommendation_types
        assert "rate_limiting" in recommendation_types
        assert "performance" in recommendation_types


class TestXGDataManager:
    """Test high-level xG data manager"""
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = XGDataManager()
        assert len(manager.providers) == 0
        assert isinstance(manager.monitor, APIHealthMonitor)
    
    def test_add_provider(self):
        """Test adding provider to manager"""
        manager = XGDataManager()
        config = ProviderConfig()
        provider = MockProvider(config)
        
        manager.add_provider("test", provider)
        assert "test" in manager.providers
    
    def test_setup_sportmonks(self):
        """Test setting up Sportmonks provider"""
        manager = XGDataManager()
        
        # Should succeed even without API key (graceful degradation)
        success = manager.setup_sportmonks()
        assert success == True
        assert "sportmonks" in manager.providers
    
    def test_setup_sportmonks_with_key(self):
        """Test setting up Sportmonks with API key"""
        manager = XGDataManager()
        
        success = manager.setup_sportmonks("test_api_key")
        assert success == True
        assert "sportmonks" in manager.providers
        assert manager.providers["sportmonks"].config.api_key == "test_api_key"
    
    def test_get_health_dashboard(self):
        """Test getting health dashboard data"""
        manager = XGDataManager()
        config = ProviderConfig()
        provider = MockProvider(config)
        manager.add_provider("test", provider)
        
        dashboard = manager.get_health_dashboard()
        
        assert "health_status" in dashboard
        assert "cost_summary" in dashboard
        assert "recommendations" in dashboard


class TestIntegration:
    """Integration tests for the complete xG data system"""
    
    @patch('airsenal.framework.xg_data_provider.session_scope')
    def test_sync_xg_data_for_gameweek(self, mock_session_scope):
        """Test syncing xG data for a gameweek"""
        # Mock database session and fixtures
        mock_session = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        # Mock fixtures
        mock_fixture = Mock()
        mock_fixture.fixture_id = 123
        mock_session.query.return_value.filter.return_value.all.return_value = [mock_fixture]
        
        # Create manager with mock provider
        manager = XGDataManager()
        config = ProviderConfig()
        provider = MockProvider(config)
        
        # Mock provider to return xG data
        xg_data = [XGDataPoint(
            player_id=456,
            fixture_id=123,
            match_date=datetime.now(),
            expected_goals=0.5
        )]
        provider.get_match_xg_data = Mock(return_value=xg_data)
        
        manager.add_provider("test", provider)
        
        # Mock update method
        manager._update_player_scores = Mock(return_value=1)
        
        result = manager.sync_xg_data_for_gameweek("2023-24", 15)
        
        assert provider.get_match_xg_data.called
        assert manager._update_player_scores.called
    
    def test_create_xg_manager_convenience_function(self):
        """Test convenience function for creating xG manager"""
        manager = create_xg_manager("test_api_key")
        
        assert isinstance(manager, XGDataManager)
        assert "sportmonks" in manager.providers


class TestErrorHandling:
    """Test error handling and graceful degradation"""
    
    def test_missing_credentials_handling(self):
        """Test handling missing API credentials"""
        # Should not raise exception when no API key provided
        provider = SportmonksProvider()
        assert provider.config.api_key is None
    
    @patch.object(SportmonksProvider, '_make_request')
    def test_api_error_handling(self, mock_request):
        """Test handling API errors"""
        provider = SportmonksProvider()
        
        # Mock API error
        mock_request.side_effect = HTTPError("API Error")
        
        # Should return empty list, not raise exception
        result = provider.get_player_xg_data(123)
        assert result == []
    
    def test_invalid_data_handling(self):
        """Test handling invalid response data"""
        transformer = DataTransformer()
        
        # Test with malformed data
        invalid_data = {"invalid": "structure"}
        result = transformer.sportmonks_to_xg_data(invalid_data, 123)
        
        # Should return empty list, not crash
        assert result == []
    
    def test_network_timeout_handling(self):
        """Test handling network timeouts"""
        config = ProviderConfig(timeout_seconds=1)
        provider = MockProvider(config)
        
        with patch('requests.Session.get') as mock_get:
            mock_get.side_effect = Timeout("Request timed out")
            
            with pytest.raises(Timeout):
                provider._make_request("https://api.test.com/endpoint")
            
            # Should record failed request
            assert provider.metrics.requests_failed > 0


class TestDashboardData:
    """Test monitoring dashboard data generation"""
    
    def test_dashboard_data_structure(self):
        """Test dashboard data has correct structure"""
        monitor = APIHealthMonitor()
        config = ProviderConfig()
        provider = MockProvider(config)
        
        provider.metrics.add_request(True, 0.5, 0.01)
        monitor.register_provider("test", provider)
        
        dashboard = monitor.export_dashboard_data()
        
        # Check required fields
        assert "health_status" in dashboard
        assert "cost_summary" in dashboard
        assert "recommendations" in dashboard
        
        # Check health status structure
        health = dashboard["health_status"]
        assert "timestamp" in health
        assert "providers" in health
        assert "test" in health["providers"]
        
        # Check cost summary structure
        costs = dashboard["cost_summary"]
        assert "total_cost" in costs
        assert "total_requests" in costs
        assert "provider_breakdown" in costs


if __name__ == "__main__":
    pytest.main([__file__])