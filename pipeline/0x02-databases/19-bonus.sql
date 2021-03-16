-- script that creates a stored procedure AddBonus that adds a new correction for a student.
DELIMITER //
CREATE PROCEDURE AddBonus(
    IN newuser_id INT,
    IN new_project VARCHAR(255),
    IN new_score INT)
    BEGIN
        IF NOT EXISTS (SELECT name FROM projects WHERE name = new_project) THEN
            INSERT INTO projects (name) VALUES (new_project);
        END IF;
       INSERT INTO corrections (user_id, project_id, score) VALUES (newuser_id, (SELECT id FROM projects WHERE name = new_project), new_score);
    END //
DELIMITER ;
