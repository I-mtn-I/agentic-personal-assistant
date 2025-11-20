#!/bin/bash
set -e

# Create target database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
    CREATE DATABASE $POSTGRES_DOCUMENTS_DB;
EOSQL

# Apply schema to target DB
echo "Applying schema to $POSTGRES_DOCUMENTS_DB..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DOCUMENTS_DB" -f /docker-entrypoint-initdb.d/schema.sql
