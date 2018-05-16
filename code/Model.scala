import org.apache.log4j.{BasicConfigurator, LogManager, Level}
import java.util.Calendar


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




trait WeakLearnerModel {
    def predict(x: List[List[Double]]): List[Double]
    def error(x: List[Double],y: List[Double],weights: List[Double]): Double
}
trait WeakLearner {
  def fit(x: List[List[Double]], y: List[Double], weights: List[Double]): WeakLearnerModel
}


object DecisionStumpModel {
  import java.util.concurrent.atomic.AtomicLong
  val id = new AtomicLong
  def getNewId:Long = {
    id.getAndIncrement
  }
}
class DecisionStumpModel(val split: Double, val left: Node = null.asInstanceOf[Node], val right: Node= null.asInstanceOf[Node]) extends WeakLearnerModel {
  import DecisionStumpModel._
  val logger = LogManager.getLogger(s"logger-${this.toString}");
  val id = DecisionStumpModel.getNewId
  def predict(x: List[List[Double]]): List[Double] = ???
  def error(x: List[Double],y: List[Double],weights: List[Double]): Double = ???
}
object DecisionStump extends WeakLearner {
  def fit(x: List[List[Double]], y: List[Double], weights: List[Double])
}
