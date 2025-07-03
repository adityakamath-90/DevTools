package com.example.calculator

/**
 * A simple calculator class that performs basic arithmetic operations.
 */
class Calculator {
    
    /**
     * Adds two integers.
     * @param a First number
     * @param b Second number
     * @return Sum of a and b
     */
    fun add(a: Int, b: Int): Int {
        return a + b
    }
    
    /**
     * Subtracts second number from first.
     * @param a First number
     * @param b Second number to subtract
     * @return Difference of a and b
     */
    fun subtract(a: Int, b: Int): Int {
        return a - b
    }
    
    /**
     * Multiplies two integers.
     * @param a First number
     * @param b Second number
     * @return Product of a and b
     */
    fun multiply(a: Int, b: Int): Int {
        return a * b
    }
    
    /**
     * Divides first number by second.
     * @param a Dividend
     * @param b Divisor
     * @return Quotient of a and b
     * @throws IllegalArgumentException if divisor is zero
     */
    fun divide(a: Int, b: Int): Double {
        if (b == 0) {
            throw IllegalArgumentException("Division by zero is not allowed")
        }
        return a.toDouble() / b.toDouble()
    }
    
    /**
     * Calculates the power of a number.
     * @param base The base number
     * @param exponent The exponent
     * @return base raised to the power of exponent
     */
    fun power(base: Double, exponent: Double): Double {
        return Math.pow(base, exponent)
    }
    
    /**
     * Calculates square root of a number.
     * @param number The number to find square root of
     * @return Square root of the number
     * @throws IllegalArgumentException if number is negative
     */
    fun sqrt(number: Double): Double {
        if (number < 0) {
            throw IllegalArgumentException("Square root of negative number is not supported")
        }
        return Math.sqrt(number)
    }
}
