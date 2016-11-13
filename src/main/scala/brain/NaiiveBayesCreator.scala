import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.transforms.Log
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import java.util.Random

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
  * Created by Lee Wan on 11/13/2016.
  */
object NaiiveBayesCreator {

  val spamFolder = "email/spam/"
  val hamFolder = "email/ham/"

  /**
    * Create predefined data for naiive bayes
    * @return
    */
  def loadDataSet(): (Array[List[String]], Array[Int]) = {
    val postingList = Array(
      List("my", "dog", "has", "flea", "problems", "help", "please"),
      List("maybe", "not", "take", "him", "to", "dog", "park", "stupid"),
      List("my", "dalmation", "is", "so", "cute", "I", "love", "him"),
      List("stop", "posting", "stupid", "worthless", "garbage"),
      List("mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"),
      List("quit", "buying", "worthless", "dog", "food", "stupid")
    )
    val classVec = Array(0,1,0,1,0,1) //1=abusive, 0=not abusive
    (postingList, classVec)
  }

  def createVocabList(dataSet:Array[List[String]]): List[String] = {
    val flattenMap = dataSet.flatMap(eachStr => eachStr)
    flattenMap.toSet.toList
  }

  //TODO: Now it is to search input words in vocabList. It should be the other way round, but then it will introduce mutable values
  def setOfWords2Vec(vocabList:List[String], inputSet:List[String]): List[Int] = {
    val returnSet = List.tabulate(vocabList.length)(n => {
      val word = vocabList(n)
      if(inputSet.contains(word)) {
        1
      }
      else {
        0
      }
    })
    returnSet
  }

  /**
    * Initial part(this may introduce underflow and consideration that 0 * anyvalues = 0
    * Result of stupid within matrix return of p1Vect is highest (probability of 16%)
    * @param trainMatrix
    * @param trainCategory
    * @return
    */
  def trainNB0(trainMatrix:Array[List[Int]], trainCategory:Array[Int]): (INDArray, INDArray, Float) = {
    val numTrainDocs  = trainMatrix.length
    val numWords = trainMatrix(0).length
    val pAbusive = trainCategory.sum / numTrainDocs.toFloat
    val p0Num = Nd4j.zeros(numWords)
    val p1Num = Nd4j.zeros(numWords)
    var p0Denom = 0.0
    var p1Denom = 0.0
    for(i<- 0 until numTrainDocs) {
      if(trainCategory(i) == 1) { //Indicated as abusive
        p1Num += trainMatrix(i).toNDArray
        p1Denom += trainMatrix(i).sum
      }
      else {
        p0Num += trainMatrix(i).toNDArray
        p0Denom += trainMatrix(i).sum
      }
    }
    val p1Vect = p1Num/p1Denom
    val p0Vect = p0Num/p0Denom
    (p0Vect, p1Vect, pAbusive)
  }

  /**
    * Enhancement for trainNB0
    * p0Num/p1Num are modified to 1 to lessen the impact that 0*n is always returning 0
    * p0Denom/p1Denom are modified to 2, as (p0Num=1) + (p1Num=1) = 2
    * p1Vect/p0Vect are modified as log, because we summing very small >0 values. ln(a*b) = ln(a) + ln(b)
    *
    * @param trainMatrix
    * @param trainCategory
    * @return
    */
  def trainNB0_Enhanced(trainMatrix:Array[List[Int]], trainCategory:Array[Int]): (INDArray, INDArray, Float) = {
    val numTrainDocs  = trainMatrix.length
    val numWords = trainMatrix(0).length
    val pAbusive = trainCategory.sum / numTrainDocs.toFloat
    val p0Num = Nd4j.ones(numWords)
    val p1Num = Nd4j.ones(numWords)
    var p0Denom = 2.0
    var p1Denom = 2.0
    for(i<- 0 until numTrainDocs) {
      if(trainCategory(i) == 1) { //Indicated as abusive
        p1Num += trainMatrix(i).toNDArray
        p1Denom += trainMatrix(i).sum
      }
      else {
        p0Num += trainMatrix(i).toNDArray
        p0Denom += trainMatrix(i).sum
      }
    }

    val p1Vect = Nd4j.getExecutioner().execAndReturn(new Log((p1Num/p1Denom)))
    val p0Vect = Nd4j.getExecutioner().execAndReturn(new Log((p0Num/p0Denom)))
    (p0Vect, p1Vect, pAbusive)
  }

  /**
    * Classify using naiive bayes algorithm
    * @param vec2Classify value to be classified
    * @param p0Vec vector value of the probability of each word occurance is
    * @param p1Vec vector value of the probability of each word occurance is
    * @param pClass1 classification probability
    * @return
    */
  def classifyNB(vec2Classify: INDArray, p0Vec:INDArray, p1Vec:INDArray, pClass1: Float): Int = {
    val p1 = (vec2Classify * p1Vec).sumNumber().doubleValue() + Math.log(pClass1)
    val p0 = (vec2Classify * p0Vec).sumNumber().doubleValue() + Math.log(1.0 - pClass1)
    if(p1 > p0) {
      return 1
    }
    else {
      return 0
    }
  }

  /**
    * Test with predifined data of accuracy.
    */
  def testingNB(): Unit = {
    val (listOPosts, listClasses) = loadDataSet()
    val myVocabList = createVocabList(listOPosts)
    val trainMat = listOPosts.map(postinDoc => setOfWords2Vec(myVocabList, postinDoc))
    val (p0V, p1V, pAb) = trainNB0_Enhanced(trainMat, listClasses)
    val testEntry = List("love", "my", "dalmation")
    val thisDoc = setOfWords2Vec(myVocabList, testEntry).toNDArray
    println(s"""[${testEntry.mkString(",")}], classifed as: ${classifyNB(thisDoc, p0V, p1V, pAb)}""")
    val testEntry2 = List("stupid", "garbage")
    val thisDoc2 = setOfWords2Vec(myVocabList, testEntry2).toNDArray
    println(s"""[${testEntry2.mkString(",")}], classifed as: ${classifyNB(thisDoc2, p0V, p1V, pAb)}""")
  }

  /**
    * Enhanced of setOfWords2Vec, count number of same occurance word in a sentence. Previous is only
    * 1 word occurance within a sentence.
    * @param vocabList
    * @param inputSet
    * @return
    */
  def bagOfWords2Vec(vocabList:List[String], inputSet:List[String]): List[Int] = {
    val returnVec = mutable.MutableList.tabulate(vocabList.length)(n=>0)
    for(word <- inputSet) {
      if(vocabList.contains(word)) {
        returnVec(vocabList.indexOf(word)) += 1
      }
    }
    returnVec.toList
  }

  def textParse(bigString: String): Array[String] = {
    val listOfToken = bigString.split("""\W""")
    listOfToken.filter(tok => tok.length > 2).map(tok => tok.toLowerCase)
  }

  /**
    * Check if the word is a spam, by comparing 2 files.
    * Probability can still be wrong due to common words like "the", "may", "for" collectively exist in both emails.
    * This can be mitigated via http://www.ranks.nl/resources/stopwords.html
    */
  def testSpam(): Unit = {
    val random = new Random()
    val docList = ListBuffer[List[String]] ()
    //val fullTextList = ListBuffer[String] ()
    val classList = ListBuffer[Int]()
    for(i <- 1 to 25) {
      val wordList = textParse(Source.fromInputStream(getClass.getResourceAsStream(spamFolder + i + ".txt")).getLines().mkString(""))
      docList += wordList.toList
      //fullTextList ++= wordList
      classList += 1
      val wordList_2 = textParse(Source.fromInputStream(getClass.getResourceAsStream(hamFolder + i + ".txt")).getLines().mkString(""))
      docList += wordList_2.toList
      //fullTextList ++= wordList
      classList += 0
    }

    val vocabList = createVocabList(docList.toArray)
    val trainingSet = ListBuffer.tabulate(50)(n=>n)
    val testSet = ListBuffer[Int]()
    for(i <- 0 until 10) {
      val randIndex = random.nextInt(trainingSet.length)
      testSet += trainingSet((randIndex))
      trainingSet -= trainingSet(randIndex)
    }

    val trainMat = new ListBuffer[List[Int]]()
    val trainClasses = ListBuffer[Int]()
    val returnVal = trainingSet.map(docIndex => {
      trainMat += bagOfWords2Vec(vocabList, docList(docIndex))
      trainClasses += classList(docIndex)
    })

    val (p0V, p1V, pSpam) = trainNB0_Enhanced(trainMat.toArray, trainClasses.toArray)
    var errorCount = 0
    testSet.foreach(docIndex => {
      val wordVector = bagOfWords2Vec(vocabList, docList(docIndex)).toNDArray
      val classification = classifyNB(wordVector, p0V, p1V, pSpam)
      if(classification != classList(docIndex)) {
        println(s"Classification ${classList(docIndex)} error: ${docList(docIndex)}")
        errorCount += 1
      }
    })
    println(s"the error rate is ${errorCount.toFloat/testSet.length}")
  }
}
