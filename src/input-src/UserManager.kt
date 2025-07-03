package com.example.user

/**
 * Represents a user in the system.
 */
data class User(
    val id: Long,
    val username: String,
    val email: String,
    var isActive: Boolean = true
)

/**
 * Service class for managing users.
 */
class UserManager {
    
    private val users = mutableListOf<User>()
    
    /**
     * Creates a new user.
     * @param username The username for the new user
     * @param email The email address for the new user
     * @return The created user
     * @throws IllegalArgumentException if username or email is empty
     */
    fun createUser(username: String, email: String): User {
        if (username.isBlank()) {
            throw IllegalArgumentException("Username cannot be empty")
        }
        if (email.isBlank()) {
            throw IllegalArgumentException("Email cannot be empty")
        }
        if (!isValidEmail(email)) {
            throw IllegalArgumentException("Invalid email format")
        }
        
        val userId = (users.maxOfOrNull { it.id } ?: 0) + 1
        val user = User(userId, username, email)
        users.add(user)
        return user
    }
    
    /**
     * Finds a user by their ID.
     * @param id The user ID to search for
     * @return The user if found, null otherwise
     */
    fun findUserById(id: Long): User? {
        return users.find { it.id == id }
    }
    
    /**
     * Updates a user's active status.
     * @param id The user ID
     * @param isActive The new active status
     * @return true if user was updated, false if user not found
     */
    fun updateUserStatus(id: Long, isActive: Boolean): Boolean {
        val user = findUserById(id)
        return if (user != null) {
            user.isActive = isActive
            true
        } else {
            false
        }
    }
    
    /**
     * Gets all active users.
     * @return List of active users
     */
    fun getActiveUsers(): List<User> {
        return users.filter { it.isActive }
    }
    
    /**
     * Deletes a user by ID.
     * @param id The user ID to delete
     * @return true if user was deleted, false if user not found
     */
    fun deleteUser(id: Long): Boolean {
        val user = findUserById(id)
        return if (user != null) {
            users.remove(user)
            true
        } else {
            false
        }
    }
    
    /**
     * Validates email format using a simple regex.
     * @param email The email to validate
     * @return true if email format is valid
     */
    private fun isValidEmail(email: String): Boolean {
        val emailRegex = "^[A-Za-z0-9+_.-]+@([A-Za-z0-9.-]+\\.[A-Za-z]{2,})$"
        return email.matches(emailRegex.toRegex())
    }
    
    /**
     * Gets the total number of users.
     * @return Total user count
     */
    fun getUserCount(): Int {
        return users.size
    }
}
