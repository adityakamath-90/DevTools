package com.example.arithmetic

import org.junit.jupiter.api.Test
import kotlin.test.*

class ArithmeticEngineTest {
    private val engine = ArithmeticEngine()

    @Test
    fun `sum should add two numbers correctly`() {
        assertEquals(12, engine.sum(7, 5))
    }

    @Test
    fun `sum with negative numbers`() {
        assertEquals(-8, engine.sum(-3, -5))
    }

    @Test
    fun `difference should subtract correctly`() {
        assertEquals(3, engine.difference(8, 5))
    }

    @Test
    fun `difference with larger second number`() {
        assertEquals(-2, engine.difference(3, 5))
    }

    @Test
    fun `product should multiply correctly`() {
        assertEquals(24, engine.product(6, 4))
    }

    @Test
    fun `product with zero should return zero`() {
        assertEquals(0, engine.product(5, 0))
    }

    @Test
    fun `quotient should divide correctly`() {
        assertEquals(3.0, engine.quotient(12, 4))
    }

    @Test
    fun `quotient with decimal result`() {
        assertEquals(2.5, engine.quotient(5, 2))
    }

    @Test
    fun `quotient by zero should throw exception`() {
        assertFailsWith<IllegalArgumentException> {
            engine.quotient(10, 0)
        }
    }

    @Test
    fun `exponentiation should calculate power correctly`() {
        assertEquals(125.0, engine.exponentiation(5.0, 3.0))
    }

    @Test
    fun `exponentiation to power of zero`() {
        assertEquals(1.0, engine.exponentiation(7.0, 0.0))
    }

    @Test
    fun `squareRoot should calculate correctly`() {
        assertEquals(4.0, engine.squareRoot(16.0))
    }

    @Test
    fun `squareRoot of one should return one`() {
        assertEquals(1.0, engine.squareRoot(1.0))
    }

    @Test
    fun `squareRoot of negative number should throw exception`() {
        assertFailsWith<IllegalArgumentException> {
            engine.squareRoot(-9.0)
        }
    }

    @Test
    fun `percentage should calculate correctly`() {
        assertEquals(25.0, engine.percentage(50, 50))
    }

    @Test
    fun `percentage of zero`() {
        assertEquals(0.0, engine.percentage(0, 100))
    }

    @Test
    fun `factorial should calculate correctly`() {
        assertEquals(120, engine.factorial(5))
    }

    @Test
    fun `factorial of zero should return one`() {
        assertEquals(1, engine.factorial(0))
    }

    @Test
    fun `factorial of negative number should throw exception`() {
        assertFailsWith<IllegalArgumentException> {
            engine.factorial(-1)
        }
    }
}
