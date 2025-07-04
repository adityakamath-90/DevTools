package com.example.calculator

import kotlin.test.*

class CalculatorTest {
    private val calculator = Calculator()

    @Test
    fun `add two integers`() {
        assertEquals(7, calculator.add(3, 4))
    }

    @Test
    fun `subtract second number from first`() {
        assertEquals(1, calculator.subtract(5, 4))
    }

    @Test
    fun `multiply two integers`() {
        assertEquals(15, calculator.multiply(3, 5))
    }

    @Test
    fun `divide first number by second`() {
        assertEquals(2.5, calculator.divide(5, 2))
    }

    @Test
    fun `calculate power of a number`() {
        assertEquals(8.0, calculator.power(2.0, 3.0))
    }

    @Test
    fun `square root of positive number`() {
        assertEquals(3.0, calculator.sqrt(9.0))
    }

    @Test
    fun `absolute value of positive number`() {
        assertEquals(5, calculator.abs(5))
    }

    @Test
    fun `modulo operation should return remainder`() {
        assertEquals(2, calculator.mod(8, 3))
    }
}