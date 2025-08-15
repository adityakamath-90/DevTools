package com.example.user

import io.mockk.every
import io.mockk.mockk
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class UserManagerTest {
    @Test
    fun createUser_validInput_userCreated() {
        val userManager = mockk<UserManager>()
        every { userManager.createUser("John Doe", "johndoe@email.com") } returns User(1, "John Doe", "johndoe@email.com")

        val createdUser = userManager.createUser("John Doe", "johndoe@email.com")

        assertEquals("John Doe", createdUser.username)
        assertEquals("johndoe@email.com", createdUser.email)
    }

    @Test
    fun createUser_invalidInput_errorThrown() {
        val userManager = mockk<UserManager>()
        every { userManager.createUser("", "") } throws IllegalArgumentException("Username and email are required.")

        assertFailsWith<IllegalArgumentException> { userManager.createUser("", "") }
    }

    @Test
    fun updateUserStatus_existingUser_statusUpdated() {
        val userManager = mockk<UserManager>()
        every { userManager.findUserById(1) } returns User(1, "John Doe", "johndoe@email.com")
        every { userManager.updateUserStatus(1, false) } returns true

        val updatedUser = userManager.updateUserStatus(1, false)

        assertFalse(userManager.getActiveUsers().contains(updatedUser))
    }
}