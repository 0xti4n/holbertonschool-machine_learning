# Databases

## Specializations - Machine Learning ― The Pipeline

## Description

* This repository contains Databases exercises

## Learning Objectives

**Understand:**

* What’s a none relational database
* What is difference between SQL and NoSQL
* How to create tables with constraints
* How to optimize queries by adding indexes
* What is and how to implement stored procedures and functions in MySQL
* What is and how to implement views in MySQL
* What is and how to implement triggers in MySQL
* What is ACID
* What is a document storage
* What are NoSQL types
* What are benefits of a NoSQL database
* How to query information from a NoSQL database
* How to insert/update/delete information from a NoSQL database
* How to use MongoDB


## Dependencies
```
Python 3.5
MySQL 5.7 (version 5.7.30)
MongoDB (version 4.2)
PyMongo (version 3.10)
```

## Repo content
* A folder named **main_sql** and **main** that contain main for some files of the following tasks:

| Task | Description |
| --- | --- |
|**0. Create a database**| script that creates the database db_0 in your MySQL server.
|**1. First table**| script that creates a table called first_table in the current database in your MySQL server.
|**2. List all in table**| Script that lists all rows of the table first_table in your MySQL server.
|**3. First add**| script that inserts a new row in the table first_table in your MySQL server.
|**4. Select the best**|  script that lists all records with a score >= 10 in the table second_table in your MySQL server.
|**5. Average**| script that computes the score average of all records in the table second_table in your MySQL server.
|**6. Temperatures #0**| script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).
|**7. Temperatures #2**| script that displays the max temperature of each state (ordered by State name).
|**8. Genre ID by show**| Write a script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.
|**9. No genre**| script that lists all shows contained in hbtn_0d_tvshows without a genre linked.
|**10. Number of shows by genre**| script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
|**11. Rotten tomatoes**| script that lists all shows from hbtn_0d_tvshows_rate by their rating.
|**12. Best genre**| script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.
|**13. We are all unique!**| Write a SQL script that creates a table users.
|**14. In and not out**| SQL script that creates a table users.
|**15. Best band ever!**| SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans.
|**16. Old school band**| SQL script that lists all bands with Glam rock as their main style, ranked by their longevity.
|**17. Buy buy buy**| SQL script that creates a trigger that decreases the quantity of an item after adding a new order.
|**18. Email validation to sent**| SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.
|**19. Add bonus**| SQL script that creates a stored procedure AddBonus that adds a new correction for a student.
|**20. Average score**| SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
|**21. Safe divide**| SQL script that creates a function SafeDiv that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.
|**22. List all databases**| Script that lists all databases in MongoDB.
|**23. Create a database**| script that creates or uses the database my_db
|**24. Insert document**| cript that inserts a document in the collection school.
|**25. All documents**| script that lists all documents in the collection school.
|**26. All matches**| script that lists all documents with name="Holberton school" in the collection school.
|**27. Count**| script that displays the number of documents in the collection school.
|**28. Update**| script that adds a new attribute to a document in the collection school.
|**29. Delete by match**| script that deletes all documents with name="Holberton school" in the collection school.
|**30. List all documents in Python**| Python function that lists all documents in a collection.
|**31. Insert a document in Python**| Python function that inserts a new document in a collection based on kwargs.
|**32. Change school topics**|  Python function that changes all topics of a school document based on the name.
|**33. Where can I learn Python?**| Python function that returns the list of school having a specific topic.
|**34. Log stats**| Python script that provides some stats about Nginx logs stored in MongoDB.
|****|
|****|
|****|
|****|
|****|
|****|
|****|
|****|
|****|
|****|
|****|
|****|

## Usage
* Clone the repo and execute the main files

## Author
- [Cristian G](https://github.com/cristian-fg)

## License
[MIT](https://choosealicense.com/licenses/mit/)
