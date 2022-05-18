DECLARE @House AS INT;

SELECT
	(
		SELECT COUNT(*)
		FROM testing.measurement m
		    JOIN testing.property p ON p.id = m.property_id
		WHERE p.house_id = @House
	) + (
		SELECT COUNT(*)
		FROM testing.rawdata r
		    JOIN testing.property p ON p.id = r.property_id
		WHERE p.house_id = @House
	) total
