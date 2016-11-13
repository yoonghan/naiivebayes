object SpamTest {
  def main(args: Array[String]): Unit = {
    val creator = NaiiveBayesCreator
    creator.testSpam
  }
}
