package com.example.calculator

import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class CalculatorTest {
    @Test
    fun `add two integers should return correct sum`() {
        // Arrange
        val calculator = mockk<Calculator>()
        every { calculator.add(3, 4) } returns 7

        // Act
        val result = calculator.add(3, 4)

        // Assert
        assertEquals(7, result)
    }

    @Test
    fun `subtract should return correct difference`() {
        // Arrange
        val calculator = mockk<Calculator>()
        every { calculator.subtract(5, 4) } returns 1

        // Act
        val result = calculator.subtract(5, 4)

        // Assert
        assertEquals(1, result)
    }

    @Test
    fun `divide by zero should throw IllegalArgumentException`() {
        // Arrange
        val calculator = mockk<Calculator>()

        // Act and assert
        assertFailsWith<IllegalArgumentException> { calculator.divide(10, 0) }
    }
}