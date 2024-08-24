"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import java.io.File
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    val config = mutableMapOf<String, Any>()

    for (arg in args) {
        if ('=' !in arg) {
            // assume it's the name of a config file
            require(!arg.startsWith("--")) { "Config file should not start with '--'" }
            val configFile = File(arg)
            println("Overriding config with ${configFile.name}:")
            println(configFile.readText())
            configFile.readLines().forEach { line ->
                val (key, value) = line.split("=")
                config[key.trim()] = parseValue(value.trim())
            }
        } else {
            // assume it's a --key=value argument
            require(arg.startsWith("--")) { "Argument should start with '--'" }
            val (key, value) = arg.substring(2).split("=")
            if (key in config) {
                val parsedValue = parseValue(value)
                require(parsedValue::class == config[key]!!::class) { "Type mismatch for key: $key" }
                println("Overriding: $key = $parsedValue")
                config[key] = parsedValue
            } else {
                System.err.println("Unknown config key: $key")
                exitProcess(1)
            }
        }
    }
}

fun parseValue(value: String): Any {
    return try {
        when {
            value.equals("true", ignoreCase = true) -> true
            value.equals("false", ignoreCase = true) -> false
            value.toIntOrNull() != null -> value.toInt()
            value.toDoubleOrNull() != null -> value.toDouble()
            else -> value
        }
    } catch (e: Exception) {
        value
    }
}
