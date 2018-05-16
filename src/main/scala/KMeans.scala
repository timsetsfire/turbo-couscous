import breeze.linalg._
import breeze.stats.distributions.Rand
import breeze.stats._
import breeze.numerics._
import breeze.stats.distributions.{Gaussian, Uniform, RandBasis}
import org.apache.log4j.{BasicConfigurator, LogManager, Level}
import java.util.Calendar

object TestData {
  val x1 = DenseMatrix.rand(10000, 2, Gaussian(0,1))
  val x2 = DenseMatrix.rand(5000, 2, Gaussian(4,3))
  val x = DenseMatrix.vertcat(x1, x2)
}


object Kmeans {
  BasicConfigurator.configure

  def goodEnough(oldCentroids: DenseMatrix[Double], centroids: DenseMatrix[Double], tolerance: Double = 1e-3): Boolean = {
    val diff = centroids - oldCentroids
    sum(norm( diff(*, ::) )) < tolerance
  }

  class CentroidInitializer(
    x: DenseMatrix[Double],
    k: Int,
    method: String = "kmeans++",
    seed: Int = java.util.Calendar.getInstance.hashCode
  ) {
    // Kmeans++ initialization
    private val (m,n) = (x.rows, x.cols)
    private val unif = Uniform(0, m)(RandBasis.withSeed(seed))
    private val centroids = DenseMatrix.zeros[Double](k, n)
    val logger = LogManager.getLogger(s"logger-${this.toString}");
    if(method=="kmeans++") {
    //  logger.info("Kmeans++ for iniatlizing cluster")
      centroids(0, ::) := x(unif.draw.toInt, ::)
      val p = DenseVector.fill(m){inf}
      val pSquared = p.copy
      val sampler = new Sampler(DenseVector.tabulate(m){i => i})
      for(i <- 1 until k) {
        val tempP = x(*, ::).map{ e => norm(centroids(i-1, ::).t - e) }
        p := min(p, tempP)
        pSquared := p*:*p

        pSquared := pSquared / sum(pSquared)
        val index = sampler.sampleWithReplacement(1,pSquared,seed).apply(0).toInt
        centroids(i, ::) := x(index, ::)
      }
    } else {
    //  logger.info("random obs for iniatlizing cluster")
      val random = Uniform(0, m)(RandBasis.withSeed(seed))
      for(i <- 0 until k) {
        centroids(i, ::) := x( random.draw.toInt, ::)
      }
    }
    def getCentroids() = this.centroids
    def getMethod() = this.method
  }

}

class Kmeans(val data: DenseMatrix[Double], val k: Int = 2,
  val method: String = "kmeans++",
  val seed: Int = Calendar.getInstance.hashCode, tolerance: Double = 1e-3) {
  import Kmeans._

  /**
  * Estimate Kmeans
  * @param data A DenseMatrix of Doubles.  Columns are features and rows are observations.
  * @param k The number of clusters.  Default Vale is 2.
  * @param init Initialize centroids via kmeans++ or random
  * @param seed A Long value used as a seed for the rng to initialized the centroids.
  * centroids are taken as random values from data
  * @param tolerance Double used to set the tolerance for improvement.
  */

  // needs a transform / predict method

  /**
  * initialize logger
  */
  val logger = LogManager.getLogger(s"logger-${this.toString}");


  val (m,p) = (data.rows, data.cols)

  /**
  * check data
  * check that number of clusters is > 1
  * check that the number of rows > number of clusters
  */
  if(k < 2) logger.warn("number of k should be integer greater than 1.")
  if(m < k) logger.warn(s"$m < $k: number of observations < number of k")

  /**
  * initialize centroids
  */

  val centroids =  new CentroidInitializer(data, k, method, seed) getCentroids

  /*
  * initialize cluster assignments
  * cluster assignments are zero indexed.
  */
  val clusterAssignment = this.transform(data)

  /*
  * inertia calcs
  * based on cluster assignments
  * calculated the sum of squares
  * returns a DenseVector of sum of squares by centroids
  */
  def inertia = {
    val squares = data(*, ::).map{ row =>
      centroids(*, ::).map{ mean =>
        val d = row - mean
        d dot d
      }
    }
    DenseVector.tabulate(k){ i =>
      val slicer = clusterAssignment :== i
      sum(squares(slicer, i))
    }
  }

  /*
  * oldCentroids is used in the fit method to terminate optimization
  */
  private val oldCentroids = DenseMatrix.zeros[Double](k, p)

  def fit(nIter: Int = 100, tolerance: Double = 1e-3): Unit = {
    /**
    * fit method for KMeans
    * @param nIter is the max number of iteratiosn to run
    * @param tolerance set the tolerance for the optimization
    */
    if(goodEnough(oldCentroids, centroids, tolerance)) {
      print(nIter)
      Unit
    }
    else if (nIter == 0) {
      print(nIter)
      Unit
    }
    else {
      oldCentroids := centroids.copy
      val norms = data(*, ::).map{ row =>
        centroids(*, ::).map{ mean => norm(row - mean)}
      }
      clusterAssignment := argmin( norms(*, ::))
      DenseVector.tabulate(k){ i =>
        val slicer = clusterAssignment :== i
        val closestPoints = data(slicer, ::)
        centroids(i, ::) := mean( closestPoints(::, *))
      }
      fit(nIter - 1, tolerance)
    }
  }

  /**
  *
  */
  def transform(x: DenseMatrix[Double]): DenseVector[Int] = {
    /**
    * transform an input dataset to their cluster assignments
    * @param x input dataset
    * @return cluster assignments
    */
    val norms = x(*, ::).map{ row =>
      centroids(*, ::).map{ mean => norm(row - mean)}
    }
    argmin(norms(*, ::))
  }
}
