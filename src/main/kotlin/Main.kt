import data.Prepare
import org.springframework.boot.CommandLineRunner
import org.springframework.stereotype.Component

@Component
class Main(private val prepare: Prepare) : CommandLineRunner {
    override fun run(vararg args: String?) {
        println("Hello World!")
        prepare.prepareShakespeare()
    }
}