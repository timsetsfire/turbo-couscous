implicit def List2Vec[T](x: List[T]) = x.toVector


// val y = List[Double](1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0)
// val x = List[Double](53,56,57,63,66,67,67,67,68,69,70,70,70,70,72,73,75,75,76,76,78,79,80,81)
//val weights = List.fill(24){1/24d}

import com.github.timsetsfire.classifier.weaklearner._
import com.github.timsetsfire.classifier._

//val x1 = List(x,x,x,x,x)

//val ab = new AdaBoost(x1, y, 10)
val ySrc = scala.io.Source.fromFile("./resources/iris_y.csv")
val y = ySrc.getLines.map{ _.toDouble}.toList

val xSrc = scala.io.Source.fromFile("./resources/iris_x.csv")

val x = xSrc.getLines.map{ _.split(",").map{ _.toDouble}.toList}.toList.transpose

// 
// val ab = new AdaBoost(x,y,500)
//
// val yhat = ab.predict(x)
//
// (yhat, y).zipped.toList.map{ case(x,y) => if(x == y) 1d else 0d}.sum / y.length.toDouble
//

//
// val x1 = x
// val x = x1(0)
// // val ab = new AdaBoost(List(x), y, 10)
//
// val weights = List.fill(y.length){1d}
// val nTotal = weights.sum
// val yTotal = (y zip weights).map{ case(y,w) => y*w}.sum
// val pTotal = yTotal / weights.sum
//
//
// /* zip data for aggregation purposes */
// /* zip data for aggregation purposes */
// val dataTuples = (x,y,weights).zipped.toList.sortWith{ _._1 < _._1 }
//
//
// /* aggregate data */
// val dataTuplesAggregated = dataTuples.foldLeft(
//   List((0d,0d,0d))
// ){
//   case (ls, next) =>
//     val (xl, yl, wl) = ls.head
//     if( xl == next._1) (xl, next._2*next._3 + yl, wl + next._3) :: ls.tail
//     else  (next._1, next._2*next._3, next._3) :: ls
// }.reverse.tail
//
// /* figure out optimal split */
// val split = dataTuplesAggregated.foldLeft(
//   List( (0d,0d,0d,0d,0d) )
// ){
//   case (ls, next) =>
//   val (x,y,w,yCuml,wCuml) = ls.head
//   (next._1, next._2, next._3, yCuml + next._2, wCuml + next._3) :: ls
// }.reverse.tail.map{
//   case(x, y, w, yCuml, wCuml) =>
//   val pLeft = yCuml / wCuml
//   val pRight = if(Math.abs(yTotal-yCuml) < 1e-15) 0d else (yTotal - yCuml)/(nTotal - wCuml)
//   (x, pLeft, pRight, (wCuml*Entropy(Vector(pLeft, 1d-pLeft))+(weights.sum-wCuml)*Entropy(Vector(pRight,1-pRight)))/weights.sum , wCuml, nTotal - wCuml)
// }.sortWith{ _._4 < _._4}
