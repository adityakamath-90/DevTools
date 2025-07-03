package com.example.string

import org.junit.jupiter.api.Test
import kotlin.test.*

class StringProcessorTest {
    private val processor = StringProcessor()

    @Test
    fun `reverse string should return correct result`() {
        assertEquals("olleh", processor.reverse("hello"))
    }

    @Test
    fun `reverse empty string should return empty string`() {
        assertEquals("", processor.reverse(""))
    }

    @Test
    fun `isPalindrome should return true for palindrome`() {
        assertTrue(processor.isPalindrome("racecar"))
    }

    @Test
    fun `isPalindrome should return false for non-palindrome`() {
        assertFalse(processor.isPalindrome("hello"))
    }

    @Test
    fun `isPalindrome should ignore case`() {
        assertTrue(processor.isPalindrome("RaceCar"))
    }

    @Test
    fun `countVowels should return correct count`() {
        assertEquals(3, processor.countVowels("hello"))
    }

    @Test
    fun `countVowels should handle empty string`() {
        assertEquals(0, processor.countVowels(""))
    }

    @Test
    fun `toUpperCase should convert correctly`() {
        assertEquals("HELLO", processor.toUpperCase("hello"))
    }

    @Test
    fun `toLowerCase should convert correctly`() {
        assertEquals("hello", processor.toLowerCase("HELLO"))
    }

    @Test
    fun `removeSpaces should remove all spaces`() {
        assertEquals("helloworld", processor.removeSpaces("hello world"))
    }

    @Test
    fun `capitalizeFirstLetter should work correctly`() {
        assertEquals("Hello", processor.capitalizeFirstLetter("hello"))
    }

    @Test
    fun `capitalizeFirstLetter should handle empty string`() {
        assertEquals("", processor.capitalizeFirstLetter(""))
    }
}
