package com.github.timsetsfire.classifier

import com.github.timsetsfire.classifier.weaklearner._

/**
  * constructor for AdaBoost
  * @constructor create a new Adaptive Boosting instance with a weak learner
  * @param x
  * @param y
  * @param k
  * @param weakLearner
  */
object AdaBoost {
  /** one hot encoder
    * the purpose of this is to expand the target vector for
    * one vs all
    * @param y target vector of doubles starting at 0 to k-1
    * @param k the total number of classes
    */
  def ohe(y: List[Double], k: Int) = {
    y.map{ elem =>
      val temp = Array.fill(k){0d}
      temp(elem.toInt) = 1d
      temp.toList
    }.transpose
  }
  /** normalize the weight vector such that it sums to 1
    * @param w an array of doubles which are weights for
    * each observation
    */
  def normW(w: Array[Double]) = {
    val z = w.sum
    for(i <- 0 until w.length) {
      w(i) = w(i) / z
    }
  }
}

class AdaBoostClassifer(x: List[List[Double]], y: List[Double], k: Int = 2) {
  import AdaBoost._

  val yOneVsAll = ohe(y, k)
  val weightOneVsAll = Array.fill(k){ Array.fill(y.length){ 1d/y.length}}

  /** fit - fit set of boosters to each class of y
    * @param nBoosts number of boosts for each category of y
    * it is currently producing a redundant set of estimates
    */
  def fit(nBoosts: Int = 2) = {
   val out = (yOneVsAll zip weightOneVsAll).par.map{
     case(y, weight) =>
     val boosts = (0 until nBoosts).par.map { boost =>
       val (ind, booster) = x.zipWithIndex.par.map{ case(elem, ind) =>  (ind, Stump.fit(elem, y, weight.toList ))}.toList.sortWith{ _._2.entropy < _._2.entropy}.head
       val err = booster.error( x(ind), y, weight.toList);
       val alpha = 1/2d * math.log( (1-err) / err) ;
       for(i <- 0 until weight.length) {
         val indicator = if( x(ind).apply(i) <= booster.split) booster.left.prediction == y(i) else booster.right.prediction == y(i)
         if(!indicator)  weight(i) = weight(i) * math.exp( alpha )
       }
       normW(weight)
       (ind, booster, alpha)
     }
    boosts.toList
  }.toList
   new AdaBoostModel(out)
 }
}

class AdaBoost(x: List[List[Double]], y: List[Double]) {
  import AdaBoost._

}


/** AdaBoostModel - is returned from the fit of the AdaBoost
  * @constructor Create an AdaBoostModel.
  * @param boosters a list of lists of boosts
  */
class AdaBoostModel[T <: DecisionStump]( val boostTriple: List[List[(Int, T, Double)]] ) {
  /** predict method
    * @param x the set of features use to create a prediction
    */
  def predict(x: List[List[Double]]) = boostTriple.map{
    boosts => boosts.map{
      case (ind, booster, alpha) =>
      booster.predict(x(ind)).map{
         e => if(e == 0d) -1d * alpha else 1d * alpha
       }
     }
   }.map{ _.transpose }.map{ _.map{ _.sum}}.transpose.map{ _.zipWithIndex.sortWith{ _._1 > _._1}.head._2 }
}
