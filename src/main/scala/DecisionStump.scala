package com.github.timsetsfire.classifier.weaklearner

import org.apache.log4j.{BasicConfigurator, LogManager, Level}
import java.util.Calendar
// import org.nd4s.Implicits._
//import org.nd4j.linalg.factory.Nd4j

//https://docs.scala-lang.org/style/scaladoc.html

/**
  * calculate the entropy given a probability distribution
  */
object Entropy{
  /** apply metho
    * @param p a prbobability distribution
    */
  def apply(p: Vector[Double]) = {
    -p.map{
      elem =>
      if(elem == 0d | elem == 1d) 0
      else elem * math.log(elem) / math.log(2)
    }.sum
  }
}

trait WeakLearner {

  def fit(x: List[Double], y: List[Double], weights: List[Double])
}


//* Stump factory for creating decision stumps

object Stump  {

  BasicConfigurator.configure
  val logger = LogManager.getLogger(s"logger-${this.toString}")

  /**
  * create a decision stump based on
  * @param x a list of double representing a feature vectors
  * @param y a list of doubles representing the (binary) target vector.
  * It is expected that y is 0 or 1, not -1 or 1.
  * @param weights a list of doubles representing the weight each observation
  * has in calculating the optimal split
  */
  def fit(x: List[Double], y: List[Double], weights: List[Double]): DecisionStump = {
    /* check inputs */
    if(x.length != weights.length | x.length != y.length) {
      logger.fatal(s"length mismatch: lengths of (x,y,w) are ${(x.length,y.length,weights.length)} respectively")
      throw new Exception(s"length mismatch lengths of (x,y,w) are ${(x.length,y.length,weights.length)} respectively")
    }

    val nTotal = weights.sum
    val yTotal = (y zip weights).map{ case(y,w) => y*w}.sum
    val pTotal = yTotal / weights.sum
    // logger.info(s" nTotal evaluates to ${nTotal}")
    // logger.info(s" yTotal evaluates to ${yTotal}")
    // logger.info(s" pTotal evaluates to ${pTotal}")

    /* zip data for aggregation purposes */
    val dataTuples = (x,y,weights).zipped.toList.sortWith{ _._1 < _._1 }

    /* aggregate data */
    val dataTuplesAggregated = dataTuples.foldLeft(
      List((0d,0d,0d))
    ){
      case (ls, next) =>
        val (xl, yl, wl) = ls.head
        if( xl == next._1) (xl, next._2*next._3 + yl, wl + next._3) :: ls.tail
        else  (next._1, next._2*next._3, next._3) :: ls
    }.reverse.tail

    /* figure out optimal split */
    val split = dataTuplesAggregated.foldLeft(
      List( (0d,0d,0d,0d,0d) )
    ){
      case (ls, next) =>
      val (x,y,w,yCuml,wCuml) = ls.head
      (next._1, next._2, next._3, yCuml + next._2, wCuml + next._3) :: ls
    }.reverse.tail.map{
      case(x, y, w, yCuml, wCuml) =>
      val pLeft = yCuml / wCuml
      val pRight = if(Math.abs(yTotal-yCuml) < 1e-15) 0d else (yTotal - yCuml)/(nTotal - wCuml)
      (x, pLeft, pRight, (wCuml*Entropy(Vector(pLeft, 1d-pLeft))+(weights.sum-wCuml)*Entropy(Vector(pRight,1-pRight)))/weights.sum , wCuml, nTotal - wCuml)
    }.sortWith{ _._4 < _._4}.head

    /* create left and right splits */
    val left = new Node(split._1, split._5, Vector(1-split._2, split._2), false)
    val right = new Node(split._1, split._6, Vector(1-split._3, split._3), true)

    /* return a decision stump */
    new DecisionStump(split._1, left, right)

  }
}


class Node(val split: Double, val weight: Double, val p: Vector[Double], gt: Boolean, left: Node = null.asInstanceOf[Node], right: Node = null.asInstanceOf[Node]) {
  /**
  * create nodes for use in decision stump
  * @param split split value
  * @param n number of records in the nodes
  * @param p probability distribution of classes with respect to node
  * @param gt boolean, if true then right split else left split
  * @param left left node based on another split of population in parent
  * @param right right node based on anaother split of the population in parent
  */
  override def toString = s" ${if(gt) ">" else "<=" } $split"
  def details = {
    f"""
${toString} \n
\tsum of weights: ${weight}\n
\tprobability distribution of target: ${p.map{elem => math.floor( math.round(elem*1000))/1000}.mkString(",")}\n
\tentropy: $entropy%2.2f
"""
  }
  val entropy = Entropy(p)
  val prediction = p.zipWithIndex.sortWith{ _._1 > _._1 }.head._2.toDouble
}


object DecisionStump {
  import java.util.concurrent.atomic.AtomicLong
  val id = new AtomicLong
  def getNewId:Long = {
    id.getAndIncrement
  }
}




/** DecisionStump -
 *  @constructor create a new DecisionStump with a split value, and two nodes
 labeled left and right.  left corresponds to <= split and right is > split
 *  @param split the value to split
 *  @param left Node containing prediction for values <= split, prob dist, and Entropy
 *  @param right Node containing prediction for values > split, prob dist, and Entropy
*/
class DecisionStump(val split: Double, val left: Node = null.asInstanceOf[Node], val right: Node= null.asInstanceOf[Node]) {
  import DecisionStump._

  val logger = LogManager.getLogger(s"logger-${this.toString}");
  val id = getNewId

  override def toString = s"id: $id \n" + left.toString + "\n" + right.toString

  def details = s"id: $id\n" + left.details + "\n" + right.details

  def predict(x: List[Double]) = {
    x.map{ elem =>
      if(elem > split) right.prediction
      else left.prediction
    }
  }

  val entropy = (left.weight * left.entropy + right.weight * right.entropy) / (left.weight + right.weight)

  /** comput the weighted error.  This can be interpretted as the probability
    * of an incorrect prediction
    * @param x a list of double representing a feature vectors
    * @param y a list of doubles representing the (binary) target vector.
    * It is expected that y is 0 or 1, not -1 or 1.
    * @param weights a list of doubles representing the weight each observation
  */
  def error(
    x: List[Double],
    y: List[Double],
    weights: List[Double]
  ) = {
    val yhat = this.predict(x)
    (yhat, y, weights).zipped.toList.map{ case(z1,z2,z3)=> if (z1==z2) 0d else z3 }.sum / weights.sum
  }
}
