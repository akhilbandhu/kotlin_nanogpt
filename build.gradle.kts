import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    id("org.springframework.boot") version "2.7.0"
    id("io.spring.dependency-management") version "1.0.11.RELEASE"
    kotlin("jvm") version "1.6.21"
    kotlin("plugin.spring") version "1.6.21"
}

group = "com.example"
version = "0.0.1-SNAPSHOT"
java.sourceCompatibility = JavaVersion.VERSION_11

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    implementation("org.springframework.boot:spring-boot-starter")
    implementation("org.jetbrains.kotlin:kotlin-reflect")
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
    implementation("com.knuddels:jtokkit:0.6.1")
    
    // DeepLearning4J and ND4J
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta7")
    implementation("org.nd4j:nd4j-native-platform:1.0.0-beta7")
    implementation("org.jetbrains.kotlinx:multik-core:0.2.1")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.1")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.5.2")

    // TensorFlow
    implementation("org.tensorflow:tensorflow-core-platform:0.5.0")
    implementation("org.tensorflow:tensorflow-core-api:0.5.0")
    
    testImplementation("org.springframework.boot:spring-boot-starter-test")

    testImplementation(kotlin("test"))
}

tasks.withType<KotlinCompile> {
    kotlinOptions {
        freeCompilerArgs = listOf("-Xjsr305=strict")
        jvmTarget = "11"
    }
}

tasks.withType<Test> {
    useJUnitPlatform()
}