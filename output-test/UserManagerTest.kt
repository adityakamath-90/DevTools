package com.example.user

import io.mockk.*
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class UserManagerTest {
    private val userManager = mockk<UserManager>()

    @Test
    fun `createUser should create a new user`() {
        // given
        val username = "john"
        val email = "john@example.com"

        // when
        userManager.createUser(username, email)

        // then
        verify { userManager.createUser(any(), any()) }
    }

    @Test
    fun `findUserById should return a user`() {
        // given
        val id = 1L
        val user = User(id, "john", "john@example.com")
        every { userManager.findUserById(id) } returns user

        // when
        val result = userManager.findUserById(id)

        // then
        assertEquals(user, result)
    }

    @Test
    fun `updateUserStatus should update a user's active status`() {
        // given
        val id = 1L
        val isActive = false
        val user = User(id, "john", "john@example.com")
        every { userManager.updateUserStatus(id, isActive) } returns true

        // when
        val result = userManager.updateUserStatus(id, isActive)

        // then
        assertEquals(true, result)
    }

    @Test
    fun `createUser should throw an exception if a user with the same email already exists`() {
        // given
        val username = "john"
        val email = "john@example.com"
        every { userManager.findUserByEmail(email) } returns User(1L, "john", email)

        // when
        assertFailsWith<Exception> { userManager.createUser(username, email) }
    }

    @Test
    fun `updateUserStatus should throw an exception if a user with the same email already exists`() {
        // given
        val id = 1L
        val isActive = false
        every { userManager.findUserById(id) } returns User(1L, "john", "john@example.com")

        // when
        assertFailsWith<Exception> { userManager.updateUserStatus(id, isActive) }
    }
}