package com.example.user

/**
 * User data class representing a user in the system.
 * @property id The unique identifier for the user
 * @property username The username for the user
 * @property email The email address for the user
 * @property isActive Whether the user is active
 */
data class User(
    val id: Long,
    val username: String,
    val email: String,
    var isActive: Boolean = true
)
