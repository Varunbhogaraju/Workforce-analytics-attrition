SELECT
    AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) AS attrition_rate
FROM employee_attrition;

SELECT
    Department,
    AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) AS attrition_rate
FROM employee_attrition
GROUP BY Department
ORDER BY attrition_rate DESC;
SELECT
    OverTime,
    COUNT(*) AS employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attritions,
    AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) AS attrition_rate
FROM employee_attrition
GROUP BY OverTime
ORDER BY attrition_rate DESC;

SELECT
    Attrition,
    AVG(MonthlyIncome) AS avg_monthly_income
FROM employee_attrition
GROUP BY Attrition;

WITH dept AS (
    SELECT
        Department,
        AVG(CASE WHEN Attrition = 'Yes' THEN 1.0 ELSE 0.0 END) AS attrition_rate
    FROM employee_attrition
    GROUP BY Department
)
SELECT
    Department,
    attrition_rate,
    RANK() OVER (ORDER BY attrition_rate DESC) AS risk_rank
FROM dept;
