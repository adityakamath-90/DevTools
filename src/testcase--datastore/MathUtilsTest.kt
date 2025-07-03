package com.example.math

import org.junit.jupiter.api.Test
import kotlin.test.*

class MathUtilsTest {
    private val mathUtils = MathUtils()

    @Test
    fun `add two integers should return correct sum`() {
        assertEquals(7, mathUtils.add(3, 4))
    }

    @Test
    fun `add negative numbers should return correct sum`() {
        assertEquals(-5, mathUtils.add(-2, -3))
    }

    @Test
    fun `subtract should return correct difference`() {
        assertEquals(1, mathUtils.subtract(5, 4))
    }

    @Test
    fun `subtract with negative result`() {
        assertEquals(-3, mathUtils.subtract(2, 5))
    }

    @Test
    fun `multiply should return correct product`() {
        assertEquals(15, mathUtils.multiply(3, 5))
    }

    @Test
    fun `multiply by zero should return zero`() {
        assertEquals(0, mathUtils.multiply(10, 0))
    }

    @Test
    fun `divide should return correct quotient`() {
        assertEquals(2.5, mathUtils.divide(5, 2))
    }

    @Test
    fun `divide by zero should throw IllegalArgumentException`() {
        assertFailsWith<IllegalArgumentException> {
            mathUtils.divide(10, 0)
        }
    }

    @Test
    fun `power should calculate correctly`() {
        assertEquals(8.0, mathUtils.power(2.0, 3.0))
    }

    @Test
    fun `power of zero should return one`() {
        assertEquals(1.0, mathUtils.power(5.0, 0.0))
    }

    @Test
    fun `square root of positive number`() {
        assertEquals(3.0, mathUtils.sqrt(9.0))
    }

    @Test
    fun `square root of zero should return zero`() {
        assertEquals(0.0, mathUtils.sqrt(0.0))
    }

    @Test
    fun `square root of negative number should throw exception`() {
        assertFailsWith<IllegalArgumentException> {
            mathUtils.sqrt(-4.0)
        }
    }

    @Test
    fun `absolute value of positive number`() {
        assertEquals(5, mathUtils.abs(5))
    }

    @Test
    fun `absolute value of negative number`() {
        assertEquals(5, mathUtils.abs(-5))
    }

    @Test
    fun `modulo operation should return remainder`() {
        assertEquals(2, mathUtils.mod(8, 3))
    }

    @Test
    fun `modulo by zero should throw exception`() {
        assertFailsWith<IllegalArgumentException> {
            mathUtils.mod(10, 0)
        }
    }
}
