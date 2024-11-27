1. Open the database:
   ```bash
   sqlite3 synthetic_data.db
   ```
2. List all tables:
   ```sql
   .tables
   ```
3. Check schema of a table:
   ```sql
   .schema table_name
   ```
4. View top 5 rows of a table:
   ```sql
   SELECT * FROM table_name LIMIT 5;
   ```
5. Exit SQLite:
   ```sql
   .exit