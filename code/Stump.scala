// import breeze.linalg._
// import breeze.stats.distributions.Rand
// import breeze.stats._
// import breeze.numerics._
// import breeze.stats.distributions.{Gaussian, Uniform, RandBasis}
import org.apache.log4j.{BasicConfigurator, LogManager, Level}
import java.util.Calendar

package classifier.weaklearner

// eslr 337

object DecisionStump {
  BasicConfigurator.configure
}
class DecisionStump {
  import DecisionStump._
  val logger = LogManager.getLogger(s"logger-${this.toString}");
}



object Stump {

  var id: Long = 0L

  def getNewId()= {
    id += 1
    id
  }

  def entropy(d: Double) = {
    if(d == 0d) 0d
    else if(d==1d) 0d
    else -( d*Math.log(d)/Math.log(2d) + (1-d)*Math.log(1-d)/Math.log(2d))
  }

  def optimalSplit(x: Array[Double], y: Array[Double], weights: Array[Double], featureIndex: Int = 0, featureName: String = "feature"): DecisionStump = {

    val nTotal = weights.sum
    val yTotal = (y zip weights).map{ case(y,w) => y*w}.sum
    val pTotal = yTotal / weights.sum
    //println(pTotal, yTotal, nTotal)

    // create the dataset - just an array of tup
    val dataTuples = x.zip(y).zip(weights).map{ case(((x,y),w)) => (x,y,w) }.sortWith{ _._1 < _._1}
    //.map{ case( ((x,y), w)) => (x,y,w) }.sortWith{ _._1 < _._1 }.map{ case(x,y, w) => (x,y,w, 0d) }

    val dataTuplesAggregated = dataTuples.foldLeft(Array((0d,0d,0d))){
      case (ls, l) =>
        val (xl, yl, wl) = ls.head
        if( xl == l._1) Array((xl, l._2*l._3 + yl, wl + l._3)) ++ ls.tail
        else Array( (l._1, l._2*l._3, l._3)) ++ ls
    }.reverse.tail

    val split = dataTuplesAggregated.foldLeft(
      Array( (0d,0d,0d,0d,0d) )
    ){
      case (ls, l) =>
      val (x,y,w,yCuml,wCuml) = ls.head
      // x, y, w, cuml y, cuml w
      Array((l._1, l._2, l._3, yCuml + l._2, wCuml + l._3)) ++ ls
    }.reverse.tail.map{
      case(x, y, w, yCuml, wCuml) =>
      val pLeft = yCuml / wCuml
      val pRight = if(Math.abs(yTotal-yCuml) < 1e-15) 0d else (yTotal - yCuml)/(nTotal - wCuml)
      (x, pLeft, pRight, (wCuml*entropy(pLeft)+(weights.sum-wCuml)*entropy(pRight))/weights.sum )
    }.sortWith{ _._4 < _._4}.head
    new DecisionStump(split._1, split._4, featureName)
  }
}

object DecisionStump {

  var id: Long = 0L

  def getNewId()= {
    id += 1
    id
  }

}
class DecisionStump(val split: Double, val entropy: Double, name: String, left: Node, right: Node) {
  import DecisionStump._
  override def toString = f"DecisionStump(name=$name, split=$split%2.2f, entropy=$entropy%2.2f)"

  def predict(x: Array[Double]) = {
    x.map{ element => if(element <= split) 1d else -1d }
  }

  def error(x: Array[Double], y: Array[Double], w: Array[Double]):Double = {
    val yhat = this.predict(x).map{ elem => if(elem > 0) 1d else 0d }
    yhat.zip(y).zip(w).map{ case((yhat,y),w) => w * (if(yhat!=y)1d else 0d)}.sum / w.sum
  }

  def error(x: Array[Double], y:Array[Double]):Double = this.error(x,y,Array.fill(y.length){1d})

}
