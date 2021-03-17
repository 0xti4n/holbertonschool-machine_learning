-- creates a stored procedure ComputeAverageWeightedScoreForUser that computes and store
-- the average weighted score for a student.
DELIMITER //
CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN newuser_id INT)
    BEGIN
        UPDATE users
        SET average_score = (SELECT SUM(score*weight) / SUM(weight)
        FROM corrections JOIN projects ON corrections.project_id = projects.id
        WHERE user_id=newuser_id)
        WHERE id = newuser_id;
    END //
DELIMITER ;
