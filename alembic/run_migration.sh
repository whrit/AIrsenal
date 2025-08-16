#!/bin/bash
# AIrsenal PlayerAttributes Migration Script
# This script safely applies the PlayerAttributes extension migration with validation checks

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log_info() {
    echo -e "${BLUE}INFO:${NC} $1"
}

log_success() {
    echo -e "${GREEN}SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1"
}

log_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Function to create backup
create_backup() {
    if [[ -n "$AIRSENAL_DB_FILE" ]]; then
        if [[ -f "$AIRSENAL_DB_FILE" ]]; then
            BACKUP_FILE="${AIRSENAL_DB_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
            log_info "Creating backup: $BACKUP_FILE"
            cp "$AIRSENAL_DB_FILE" "$BACKUP_FILE"
            log_success "Backup created successfully"
            echo "$BACKUP_FILE"
        else
            log_error "Database file not found: $AIRSENAL_DB_FILE"
            exit 1
        fi
    else
        log_warning "AIRSENAL_DB_FILE not set - using default location"
        # Try to find the default database location
        if [[ -n "$AIRSENAL_HOME" ]]; then
            DEFAULT_DB="$AIRSENAL_HOME/data.db"
        else
            DEFAULT_DB="$HOME/.airsenal/data.db"
        fi
        
        if [[ -f "$DEFAULT_DB" ]]; then
            BACKUP_FILE="${DEFAULT_DB}.backup.$(date +%Y%m%d_%H%M%S)"
            log_info "Creating backup: $BACKUP_FILE"
            cp "$DEFAULT_DB" "$BACKUP_FILE"
            log_success "Backup created successfully"
            echo "$BACKUP_FILE"
        else
            log_error "Database file not found: $DEFAULT_DB"
            exit 1
        fi
    fi
}

# Function to run validation
run_validation() {
    local validation_type=$1
    log_info "Running $validation_type validation..."
    
    cd "$PROJECT_ROOT"
    if python alembic/migration_validation.py "$validation_type"; then
        log_success "$validation_type validation passed"
        return 0
    else
        log_error "$validation_type validation failed"
        return 1
    fi
}

# Function to run migration
run_migration() {
    log_info "Applying migration..."
    cd "$PROJECT_ROOT"
    
    if alembic upgrade head; then
        log_success "Migration applied successfully"
        return 0
    else
        log_error "Migration failed"
        return 1
    fi
}

# Function to show help
show_help() {
    cat << EOF
AIrsenal PlayerAttributes Migration Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -d, --dry-run       Run pre-migration checks only (no migration)
    -f, --force         Skip confirmation prompts
    -s, --skip-backup   Skip backup creation (not recommended)
    -r, --rollback      Rollback the migration

Examples:
    $0                  # Full migration with all safety checks
    $0 --dry-run        # Check if migration is safe to run
    $0 --force          # Run migration without confirmation
    $0 --rollback       # Rollback the migration

Environment Variables:
    AIRSENAL_DB_FILE    Path to database file (if not using default)
    AIRSENAL_HOME       AIrsenal home directory

EOF
}

# Parse command line arguments
DRY_RUN=false
FORCE=false
SKIP_BACKUP=false
ROLLBACK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -s|--skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        -r|--rollback)
            ROLLBACK=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "=============================================="
    echo "  AIrsenal PlayerAttributes Migration"
    echo "=============================================="
    echo
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/alembic.ini" ]]; then
        log_error "alembic.ini not found. Please run from AIrsenal project root."
        exit 1
    fi
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    if [[ "$ROLLBACK" == true ]]; then
        # Rollback mode
        log_warning "ROLLBACK MODE - This will remove all new PlayerAttributes columns!"
        
        if [[ "$FORCE" != true ]]; then
            echo
            read -p "Are you sure you want to rollback? This will cause data loss! (type 'yes' to confirm): " confirm
            if [[ "$confirm" != "yes" ]]; then
                log_info "Rollback cancelled"
                exit 0
            fi
        fi
        
        # Check rollback safety
        if ! run_validation "rollback-check"; then
            log_warning "Rollback safety check detected potential data loss"
            if [[ "$FORCE" != true ]]; then
                echo
                read -p "Continue with rollback anyway? (type 'yes' to confirm): " confirm
                if [[ "$confirm" != "yes" ]]; then
                    log_info "Rollback cancelled"
                    exit 0
                fi
            fi
        fi
        
        # Create backup before rollback
        if [[ "$SKIP_BACKUP" != true ]]; then
            BACKUP_FILE=$(create_backup)
            log_info "Backup location: $BACKUP_FILE"
        fi
        
        # Run rollback
        log_info "Running rollback..."
        if alembic downgrade -1; then
            log_success "Rollback completed successfully"
        else
            log_error "Rollback failed"
            exit 1
        fi
        
        exit 0
    fi
    
    # Normal migration mode
    log_info "Migration: extend_player_attributes_with_ml_features"
    log_info "Adds 18 new columns and 8 performance indexes to PlayerAttributes table"
    echo
    
    # Run pre-migration validation
    if ! run_validation "pre"; then
        log_error "Pre-migration validation failed. Please fix issues before proceeding."
        exit 1
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_success "Dry run completed - migration appears safe to execute"
        exit 0
    fi
    
    # Create backup
    if [[ "$SKIP_BACKUP" != true ]]; then
        BACKUP_FILE=$(create_backup)
        log_info "Backup location: $BACKUP_FILE"
    else
        log_warning "Skipping backup creation - THIS IS RISKY!"
    fi
    
    # Confirmation prompt
    if [[ "$FORCE" != true ]]; then
        echo
        log_warning "This migration will modify your database schema by adding 18 columns and 8 indexes."
        read -p "Do you want to proceed? (y/N): " confirm
        if [[ "$confirm" != [yY] ]]; then
            log_info "Migration cancelled"
            exit 0
        fi
    fi
    
    # Run migration
    if ! run_migration; then
        log_error "Migration failed. Your data is safe in the backup: $BACKUP_FILE"
        log_info "You can restore with: cp \"$BACKUP_FILE\" \"\$AIRSENAL_DB_FILE\""
        exit 1
    fi
    
    # Run post-migration validation
    if ! run_validation "post"; then
        log_error "Post-migration validation failed. Consider rollback."
        log_info "Backup location: $BACKUP_FILE"
        exit 1
    fi
    
    echo
    log_success "Migration completed successfully!"
    log_info "Backup saved to: $BACKUP_FILE"
    log_info "New PlayerAttributes features are now available for ML predictions"
    echo
    echo "Next steps:"
    echo "  1. Update your feature computation pipelines to populate new columns"
    echo "  2. Modify ML models to utilize new features"
    echo "  3. Test FPL predictions with enhanced data"
}

# Run main function
main "$@"