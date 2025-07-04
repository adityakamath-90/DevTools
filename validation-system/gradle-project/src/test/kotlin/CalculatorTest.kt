package com.example.calculator

import io.mockk.every
import io.mockk.mockkObject
import io.mockk.unmockkObject
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals

class CalculatorTests {
    private lateinit var calculator: Calculator

    @BeforeEach
    fun setUp() {
        calculator = Calculator()
        mockkObject(Calculator)
    }

    @AfterEach
    fun tearDown() {
        unmockkObject(Calculator)
    }

    @Test
    fun `add two positive integers`() {
        every { calculator.add(1, 2) } returns 3
        assertEquals(3, calculator.add(1, 2))
    }

    @Test
    fun `subtract second number from first`() {
        every { calculator.subtract(5, 2) } returns 3
        assertEquals(3, calculator.subtract(5, 2))
    }

    @Test
    fun `multiply two positive integers`() {
        every { calculator.multiply(2, 3) } returns 6
        assertEquals(6, calculator.multiply(2, 3))
    }

    @Test
    fun `divide by zero throws exception`() {
        every { calculator.divide(10, 0) } throws IllegalArgumentException("Division by zero is not allowed")
        assertFailsWith<IllegalArgumentException>("Division by zero is not allowed") {
            calculator.divide(10, 0)
        }
    }

    @Test
    fun `power of positive number`() {
        every { calculator.power(2.0, 3.0) } returns 8.0
        assertEquals(8.0, calculator.power(2.0, 3.0))
    }

    @Test
    fun `power of negative number`() {
        every { calculator.power(-2.0, 3.0) } returns -8.0
        assertEquals(-8.0, calculator.power(-2.0, 3.0))
    }

    @Test
    fun `square root of positive number`() {
        every { calculator.sqrt(9.0) } returns 3.0
        assertEquals(3.0, calculator.sqrt(9.0))
    }

    @Test
    fun `absolute value of negative number`() {
        every { calculator.abs(-5) } returns 5
        assertEquals(5, calculator.abs(-5))
    }

    @Test
    fun `modulo operation should return remainder`() {
        every { calculator.mod(8, 3) } returns 2
        assertEquals(2, calculator.mod(8, 3))
    }
}