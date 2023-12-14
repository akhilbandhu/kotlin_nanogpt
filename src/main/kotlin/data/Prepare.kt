package data

import java.io.File
import java.net.URL
import java.net.HttpURLConnection
import com.knuddels.jtokkit.Encodings.newDefaultEncodingRegistry
import com.knuddels.jtokkit.api.ModelType

class Prepare {
    fun prepareShakespeare() {
        val currentDir = File("").absolutePath
        val inputFilePath = "$currentDir/shakespeare.txt"

        // Download the dataset if it does not exist
        if (!File(inputFilePath).exists()) {
            val dataUrl = URL("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
            val connection = dataUrl.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"

            val data = connection.inputStream.bufferedReader().use { it.readText() }
            File(inputFilePath).writeText(data)

            connection.disconnect()
        }

        // Read the dataset
        val data = File(inputFilePath).readText()
        val n = data.length
        println("length of dataset: $n")
        println("number of characters in dataset: ${data.toSet().sorted().joinToString("")}")
        println("vocab_size: ${data.toSet().sorted().size}")
        val trainData = data.substring(0, (n * 0.9).toInt())
        val valData = data.substring((n * 0.9).toInt())

        // Tokenization using JTokkit
        val encodingRegistry = newDefaultEncodingRegistry()
        val encoding = encodingRegistry.getEncodingForModel(ModelType.TEXT_DAVINCI_003)
        val trainIds = encoding.encodeOrdinary(trainData) // Tokenize data
        val valIds = encoding.encodeOrdinary(valData)
        println("train has ${trainIds.size} tokens")
        println("val has ${valIds.size} tokens")

        // Writing to binary files
        writeTokensToFile("$currentDir/train.bin", trainIds)
        writeTokensToFile("$currentDir/val.bin", valIds)
    }

    // Write tokens to a binary file
    private fun writeTokensToFile(filename: String, tokens: List<Int>) {
        File(filename).outputStream().use { fos ->
            tokens.forEach { token ->
                fos.write((token shr 8) and 0xFF)
                fos.write(token and 0xFF)
            }
        }
    }
}