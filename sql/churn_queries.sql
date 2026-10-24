-- Customer churn analysis: SQL queries
-- Database: telco.db (SQLite), table: customers
-- The risk_actions table is created in notebooks/07_sql_analysis.ipynb

-- 1. Overall churn rate
SELECT COUNT(*) AS total_customers,
       SUM(Churn) AS churned,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- 2. Customers paying more than 100 a month
SELECT COUNT(*) AS customers,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
WHERE MonthlyCharges > 100;

-- 3. Churn by contract type
SELECT Contract,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 4. Churn by tenure group (CASE)
SELECT CASE
         WHEN tenure <= 12 THEN '1) 0-12 months'
         WHEN tenure <= 24 THEN '2) 13-24 months'
         WHEN tenure <= 48 THEN '3) 25-48 months'
         ELSE '4) 49-72 months'
       END AS tenure_group,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY tenure_group;

-- 5. Average monthly spending by internet service
SELECT InternetService,
       ROUND(AVG(MonthlyCharges), 1) AS avg_monthly_charge
FROM customers
GROUP BY InternetService;

-- 6. Customer groups with the highest churn (HAVING filters small groups)
SELECT Contract, InternetService,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Contract, InternetService
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC
LIMIT 5;

-- 7. Customers paying more than the average (subquery)
SELECT COUNT(*) AS customers,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers
WHERE MonthlyCharges > (SELECT AVG(MonthlyCharges) FROM customers);

-- 8. Each contract's share of all churners (window function)
SELECT Contract,
       SUM(Churn) AS churned,
       ROUND(100.0 * SUM(Churn) / SUM(SUM(Churn)) OVER (), 1) AS pct_of_all_churners
FROM customers
GROUP BY Contract
ORDER BY churned DESC;

-- 9. Top 3 highest-paying customers in each contract type (ROW_NUMBER)
SELECT Contract, customerID, MonthlyCharges, Churn, rank_in_contract
FROM (
    SELECT Contract, customerID, MonthlyCharges, Churn,
           ROW_NUMBER() OVER (PARTITION BY Contract
                              ORDER BY MonthlyCharges DESC) AS rank_in_contract
    FROM customers
)
WHERE rank_in_contract <= 3
ORDER BY Contract, rank_in_contract;

-- 10. JOIN customers with suggested actions per risk group
SELECT c.risk_category,
       a.suggested_action,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(c.Churn) / COUNT(*), 1) AS churn_rate_pct
FROM customers AS c
INNER JOIN risk_actions AS a
    ON c.risk_category = a.risk_category
GROUP BY c.risk_category, a.suggested_action
ORDER BY churn_rate_pct DESC;