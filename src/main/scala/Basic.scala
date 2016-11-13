object Basic {
  def main(args: Array[String]): Unit = {
    val creator = NaiiveBayesCreator
    val (listOPost, listClasses) = creator.loadDataSet()
    val myVocabList = creator.createVocabList(listOPost)
    println(s"""Vocabulary available:[ ${myVocabList.mkString(",")}]""")
    println(s"""Vector of first sentence:[ ${creator.setOfWords2Vec(myVocabList, listOPost(0))}""")
    println(s"""Vector of second sentence:[ ${creator.setOfWords2Vec(myVocabList, listOPost(3))}""")
    println("------Creation Completed-----")

    val trainMat = listOPost.map(postinDoc => creator.setOfWords2Vec(myVocabList, postinDoc))
    val (p0V, p1V, pAB) = creator.trainNB0(trainMat, listClasses)

    println(s"Abusive word: ${pAB}")
    println("p0V:" + p0V)
    println("p1V:" + p1V)

    creator.testingNB()
  }
}
