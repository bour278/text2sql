# SQL SQLite Cheat Sheet for Easier Applications

This cheat sheet covers common SQL queries and operations for SQLite, optimized for quick lookups and practical applications.

---

## **Basic Database Operations**

### **Creating a Database**
```sql
-- Create a new SQLite database
sqlite3 mydatabase.db
```

### **Creating a Table**
```sql
CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    join_date TEXT
);
```

### **Inserting Data into a Table**
```sql
INSERT INTO customers (name, email, join_date)
VALUES ('John Doe', 'johndoe@example.com', '2024-01-01');
```

---

## **Basic Queries**

### **Selecting All Data from a Table**
```sql
SELECT * FROM customers;
```

### **Selecting Specific Columns**
```sql
SELECT name, email FROM customers;
```

### **Filtering Data with WHERE**
```sql
SELECT * FROM customers
WHERE join_date > '2023-12-31';
```

### **Sorting Data with ORDER BY**
```sql
SELECT * FROM customers
ORDER BY join_date DESC;
```

### **Limiting the Results**
```sql
SELECT * FROM customers
LIMIT 5;
```

---

## **Aggregate Functions**

### **Counting Rows**
```sql
SELECT COUNT(*) FROM customers;
```

### **Calculating Average**
```sql
SELECT AVG(age) FROM customers;
```

### **Sum of a Column**
```sql
SELECT SUM(total_purchase) FROM orders;
```

### **Grouping Data**
```sql
SELECT join_date, COUNT(*) AS total_customers
FROM customers
GROUP BY join_date;
```

### **Filtering with HAVING**
```sql
SELECT join_date, COUNT(*) AS total_customers
FROM customers
GROUP BY join_date
HAVING total_customers > 5;
```

---

## **Joining Tables**

### **Inner Join**
```sql
SELECT customers.name, orders.total_amount
FROM customers
INNER JOIN orders ON customers.id = orders.customer_id;
```

### **Left Join**
```sql
SELECT customers.name, orders.total_amount
FROM customers
LEFT JOIN orders ON customers.id = orders.customer_id;
```

---

## **Updating Data**

### **Updating a Single Row**
```sql
UPDATE customers
SET email = 'newemail@example.com'
WHERE id = 1;
```

### **Updating Multiple Rows**
```sql
UPDATE customers
SET status = 'active'
WHERE join_date < '2024-01-01';
```

---

## **Deleting Data**

### **Deleting a Single Row**
```sql
DELETE FROM customers
WHERE id = 1;
```

### **Deleting All Data from a Table**
```sql
DELETE FROM customers;
```

### **Dropping a Table**
```sql
DROP TABLE IF EXISTS customers;
```

---

## **Creating and Using Indexes**

### **Creating an Index**
```sql
CREATE INDEX idx_name ON customers (name);
```

### **Using Indexes**
```sql
SELECT * FROM customers
WHERE name = 'John Doe';  -- This will use the index created above
```

---

## **Subqueries**

### **Simple Subquery**
```sql
SELECT name, email
FROM customers
WHERE id = (SELECT customer_id FROM orders WHERE total_amount > 100);
```

### **Subquery with IN**
```sql
SELECT name, email
FROM customers
WHERE id IN (SELECT customer_id FROM orders WHERE total_amount > 100);
```

---

## **Transactions**

### **Starting a Transaction**
```sql
BEGIN TRANSACTION;
```

### **Committing a Transaction**
```sql
COMMIT;
```

### **Rolling Back a Transaction**
```sql
ROLLBACK;
```

---

## **Working with Dates**

### **Getting the Current Date**
```sql
SELECT DATE('now');
```

### **Formatting Date**
```sql
SELECT strftime('%Y-%m-%d', '2024-01-01');
```

### **Date Difference**
```sql
SELECT julianday('2024-01-01') - julianday('2023-12-31');
```

---

## **String Functions**

### **Concatenating Strings**
```sql
SELECT name || ' ' || email AS customer_info FROM customers;
```

### **Changing Case**
```sql
SELECT UPPER(name) FROM customers;  -- Converts name to uppercase
SELECT LOWER(name) FROM customers;  -- Converts name to lowercase
```

### **Replacing Substrings**
```sql
SELECT REPLACE(email, 'example.com', 'newdomain.com') FROM customers;
```

---

## **Foreign Keys**

### **Enabling Foreign Key Constraints**
```sql
PRAGMA foreign_keys = ON;
```

### **Creating a Foreign Key**
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER,
    total_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
```

---

## **Backup and Restore**

### **Backing Up a Database**
```bash
sqlite3 mydatabase.db .backup mydatabase_backup.db
```

### **Restoring a Database**
```bash
sqlite3 mydatabase_backup.db .restore mydatabase.db
```

---

## **Importing & Exporting Data**

### **Import CSV into Table**
```sql
.mode csv
.import 'data.csv' customers
```

### **Export Table to CSV**
```sql
.headers on
.mode csv
.output 'customers_data.csv'
SELECT * FROM customers;
```

---

## **Optimizing Queries**

### **EXPLAIN QUERY PLAN**
```sql
EXPLAIN QUERY PLAN SELECT * FROM customers WHERE name = 'John Doe';
```

---

## **SQLite-Specific Commands**

### **Viewing Database Schema**
```sql
.schema
```

### **Listing Tables**
```sql
.tables
```

### **Viewing Table Structure**
```sql
PRAGMA table_info(customers);
```

---

## **Documentation Reference**

For more detailed usage and advanced functionality, refer to the official SQLite documentation: [https://www.sqlite.org/docs.html](https://www.sqlite.org/docs.html).

