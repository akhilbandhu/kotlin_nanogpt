package model

import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer
import org.deeplearning4j.nn.params.DefaultParamInitializer
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.inputs.InputType


/**
 * ChatGPT trying to explain this to a 5 yr old:
 * Think of layer normalization like a game where everyone gets a turn to speak equally.
 * In a computer's brain, when it's learning or deciding something, some parts may talk too loudly or too softly.
 * Layer normalization is like a rule in the game that helps every part speak at a nice, even level, so the computer can understand things better and make smarter choices.
 * It's like making sure everyone in a circle gets to share their story without anyone being too loud or too quiet.
 *
 * One of the steps in a Transformer block is to apply layer normalization to this matrix.
 * This is an operation that normalizes the values in each column of the matrix separately.
 *
 * @param nDim Number of dimensions to the layer
 * @param bias If you need to include bias or not
 */

class LayerNormalization(
    private val nDim: Int,
    private val bias: Boolean
) : SameDiffLayer() {
    /**
     * In this method, you define the parameters and their shapes
     * For this layer, we have a weight matrix (nIn x nOut) plus a bias array
     * @param params A helper class that allows you to define the parameters
     */
    override fun defineParameters(params: SDLayerParams) {
        params.addWeightParam(DefaultParamInitializer.GAIN_KEY, nDim.toLong())
        params.addBiasParam(DefaultParamInitializer.BIAS_KEY, 1, nDim.toLong())
    }

    /**
     * In the defineLayer method, you define the actual layer forward pass
     * For this layer, we are returning out = activationFn( input*weights + bias)
     *
     * @param sd         The SameDiff instance for this layer
     * @param layerInput A SDVariable representing the input activations for the layer
     * @param paramTable A map of parameters for the layer. These are the SDVariables corresponding to whatever you defined
     * in the defineParameters method
     * @return
     */
    override fun defineLayer(
        sd: SameDiff,
        layerInput: SDVariable,
        paramTable: Map<String, SDVariable>,
        mask: SDVariable?
    ): SDVariable {
        val weights = paramTable[DefaultParamInitializer.WEIGHT_KEY]
        val biases = if (bias) paramTable[DefaultParamInitializer.BIAS_KEY] else null

        // Using SameDiff's layerNorm function
        return sd.nn.layerNorm("layerNorm", layerInput, weights, biases, false, nDim)
    }

    /**
     * This method is used to initialize the parameter.
     * For example, we are setting the bias parameter to 0, and using the specified DL4J weight initialization type
     * for the weights
     * @param params Map of parameters. These are the INDArrays corresponding to whatever you defined in the
     * defineParameters method
     */
    override fun initializeParameters(params: Map<String, INDArray>) {
        params[DefaultParamInitializer.BIAS_KEY]!!.assign(0)
        initWeights(nDim, nDim, weightInit, params[DefaultParamInitializer.WEIGHT_KEY])
    }


    override fun getOutputType(layerIndex: Int, inputType: InputType): InputType {
        // Assuming the output type is the same as the input type
        return inputType
    }
}