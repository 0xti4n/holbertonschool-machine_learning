-- trigger that resets the attribute valid_email only when the email has been changed.
DELIMITER //
CREATE TRIGGER upd_email
  BEFORE UPDATE ON users
  FOR EACH ROW
BEGIN
  IF STRCMP(NEW.email, OLD.email) != 0 THEN
    SET NEW.valid_email = 0;
  END IF;
END//
DELIMITER ;
