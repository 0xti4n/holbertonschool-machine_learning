-- function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.
DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT)
    RETURNS FLOAT
    BEGIN
        DECLARE div_value FLOAT;
        SET div_value = 0;
        IF (a != 0 AND b != 0) THEN
            SET div_value = a / b;
        END IF;
    RETURN div_value;
    END //
DELIMITER ;
