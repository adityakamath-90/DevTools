package com.example.user

import io.mockk.every
import io.mockk.impl.annotations.MockK
import io.mockk.junit5.MockKExtension
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.extension.ExtendWith
import java.lang.IllegalArgumentException
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@ExtendWith(MockKExtension::class)
internal class UserManagerTest {
    @MockK
    private lateinit var users: MutableList<User>

    @Test
    fun `create user should return created user`() {
        // given
        val username = "username"
        val email = "email@example.com"

        every { users.add(any()) } returns true

        // when
        val user = UserManager().createUser(username, email)

        // then
        assertEquals(1L, user.id)
        assertEquals(username, user.username)
        assertEquals(email, user.email)
        assertTrue(user.isActive)
    }

    @Test
    fun `create user should throw IllegalArgumentException for empty username`() {
        // given
        val username = ""
        val email = "email@example.com"

        every { users.add(any()) } returns true

        // when, then
        assertFailsWith<IllegalArgumentException>("Username cannot be empty") {
            UserManager().createUser(username, email)
        }
    }

    @Test
    fun `create user should throw IllegalArgumentException for empty email`() {
        // given
        val username = "username"
        val email = ""

        every { users.add(any()) } returns true

        // when, then
        assertFailsWith<IllegalArgumentException>("Email cannot be empty") {
            UserManager().createUser(username, email)
        }
    }

    @Test
    fun `find user by id should return user`() {
        // given
        val id = 1L
        every { users.firstOrNull { it.id == id } } returns User(1L, "username", "email@example.com")

        // when
        val user = UserManager().findUserById(id)

        // then
        assertEquals(id, user?.id)
        assertEquals("username", user?.username)
        assertEquals("email@example.com", user?.email)
    }

    @Test
    fun `find user by id should return null for non-existent id`() {
        // given
        val id = 1L
        every { users.firstOrNull { it.id == id } } returns null

        // when
        val user = UserManager().findUserById(id)

        // then
        assertEquals(null, user)
    }
}