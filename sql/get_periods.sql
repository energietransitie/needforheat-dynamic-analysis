DECLARE @House AS INT;

-- Note that the operator is to compare the value is set to = but can be modified in the code
DECLARE @Value AS FLOAT;
DECLARE @Parameter AS TEXT;

-- Note that the start and end datetime can be removed to filter all data
DECLARE @Start AS DATETIME;
DECLARE @End AS DATETIME;

-- Resulting in periods in which the given condition is continuously TRUE
SELECT * 
FROM (
    SELECT 
        timestamp AS 'date_start', 
        LEAD(prev_timestamp) OVER(ORDER BY - i DESC) AS 'date_end' 
    FROM (
        SELECT 
            i, 
            CASE WHEN value = @Value THEN NULL ELSE prev_timestamp END AS 'prev_timestamp', 
            CASE WHEN prev_value = @Value THEN NULL ELSE timestamp END AS 'timestamp', 
            prev_value, 
            value 
        FROM (
            SELECT 
                ROW_NUMBER() OVER(ORDER BY timestamp) AS 'i', 
                LAG(m.timestamp) OVER(ORDER BY timestamp) AS 'prev_timestamp', 
                m.timestamp AS 'timestamp', 
                LAG(m.value) OVER(ORDER BY timestamp) AS 'prev_value', 
                m.value AS 'value' 
            FROM measurement m 
                JOIN property p ON p.id = m.property_id 
                JOIN parameter d ON d.id = p.parameter_id
            WHERE p.house_id = @House AND 
                d.name = @Parameter AND 
                m.timestamp >= @Start AND 
                m.timestamp <= @End 
            ORDER BY m.timestamp
        ) t
        WHERE (
            value = @Value AND (
                prev_value IS NULL OR NOT (prev_value = @Value))
            ) OR (
                prev_value = @Value AND NOT (value = @Value)
            ) 
        UNION 
        SELECT 
            NULL AS 'i', 
            (
                SELECT 
                    MAX(timestamp) 
                FROM measurement m 
                    JOIN property p ON p.id = m.property_id 
                    JOIN parameter d ON d.id = p.parameter_id
                WHERE 
                    p.house_id = @House AND 
                    d.name = @Parameter AND 
                    m.timestamp >= @Start AND 
                    m.timestamp <= @End
            ) AS 'prev_timestamp', 
            NULL AS 'timestamp', 
            0 AS 'prev_value', 
            0 AS 'value'
    ) t2
) t1 
WHERE 
    date_start IS NOT NULL AND 
    date_end IS NOT NULL AND
    date_end > date_start;

-- Note that the last line can be changed to date_end >= date_start to include 'periods' with a singular measurement
