-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(IN newuser_id INT)
    BEGIN
        UPDATE users
        SET average_score = (SELECT AVG(score) FROM corrections WHERE user_id = newuser_id)
        WHERE id = newuser_id;
    END //
DELIMITER ;
