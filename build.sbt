lazy val root = (project in file(".")).
  settings(
    name := "NaiiveBayeswSbt",
    version := "1.0",
    scalaVersion := "2.11.8"
)

libraryDependencies ++= Seq(
  "org.nd4j" % "nd4s_2.11" % "0.6.0",
  "org.nd4j" % "nd4j-native-platform" % "0.6.0",
  "org.slf4j" % "slf4j-simple" % "1.7.21"
)
