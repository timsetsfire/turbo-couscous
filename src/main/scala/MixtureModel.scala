import breeze.linalg._
import breeze.stats.distributions._
import breeze.stats._
import breeze.numerics._


object GaussianMixtureModel {

  // "full" (each component has its own general covariance matrix), shape of (nMixtures, x.cols, x.cols)
  // "tied" (all components share the same general covariance matrix), shape of (x.cols, x.cols)
  // "diag" (each component has its own diagonal covariance matrix), shape of (nMixtures, x.cols)
  // "spherical" (each component has its own single variance). shape of (nMixtures,)
  def fullMeans(x: DenseMatrix[Double], resp: DenseMatrix[Double]) = {
    val temp = resp.t * x
    val sumResp = sum(resp(::, *)).t
    temp(::, *).map{ col => col / sumResp}
  }
  def tiedMeans(x: DenseMatrix[Double], resp: DenseMatrix[Double]) = {
    mean(x(::, *)).t
  }
  def singleMean(x: DenseMatrix[Double], resp: DenseMatrix[Double]) = ???
  def fullCov(x: DenseMatrix[Double], means: DenseMatrix[Double], resp: DenseMatrix[Double], nMixtures: Int,  covReg: Double) = {
    val diffx = means(*, ::).map{ meanRow =>
      x(*, ::).map{ xRow => xRow - meanRow}
    }
    val covs = DenseVector.tabulate(nMixtures){ i =>
        val px = diffx.apply(i)(::, *).map{ col => col *:* resp(::, i) }
        px.t * diffx.apply(i) / sum(resp(::, i))
    }
    covs.map{ covMat =>
      diag(covMat) := diag(covMat) + covReg
      covMat
    }
  }
  def sphericalCov(x: DenseMatrix[Double], means: DenseMatrix[Double], resp: DenseMatrix[Double], nMixtures: Int, covReg: Double)  = {
    val cov = diagCov(x, means, resp, nMixtures, covReg)
    cov.map{ v =>
      val temp = mean(v) * DenseMatrix.eye[Double](x.cols)
      diag(temp) := diag(temp) + covReg
      temp
    }
  }
  def diagCov(x: DenseMatrix[Double], means: DenseMatrix[Double], resp: DenseMatrix[Double], nMixtures: Int, covReg: Double) = {
    val diffx = means(*, ::).map{ meanRow =>
      x(*, ::).map{ xRow => xRow - meanRow}
    }
    val covs = DenseVector.tabulate(nMixtures){ i =>
          val sq =  diffx.apply(i)(::, *).map{ col => (col *:* resp(::, i)) *:* col}
          val temp = diag(sum(sq(::, *)).t / sum(resp(::, i)))
          diag(temp) := diag(temp) + covReg
          temp
      }
    covs
  }
  def tiedCov(x: DenseMatrix[Double], means: DenseMatrix[Double],  resp: DenseMatrix[Double],  nMixtures: Int, covReg: Double) = {
    // means is shape (nMixtures, x.cols)
    //
    val sumResp = sum(resp(::, *)).t
    val xTx = x.t * x
    val means2 = means(::, *).map{ col => col *:* sumResp}.t * means
    val covariance = xTx - means2
    covariance := 1/sum(sumResp) * covariance
    diag(covariance) := diag(covariance) + covReg
    covariance
  }
  def mStep(x: DenseMatrix[Double], resp: DenseMatrix[Double]) = ???
}

class GaussianMixtureModel(
  val x: DenseMatrix[Double],
  val nMixtures: Int,
  val covarianceType: String = "full",
  val regCovariance: Double = 0.01,
  val initParams: String = "random",
  var mixtureProbsInit: DenseMatrix[Double] = null.asInstanceOf[DenseMatrix[Double]],
  var meansInit: DenseMatrix[Double] = null.asInstanceOf[DenseMatrix[Double]],
  var precisionsInit: DenseMatrix[Double] = null.asInstanceOf[DenseMatrix[Double]], // inverse of cov
  val warmStart: Boolean = true
) {
  import GaussianMixtureModel._
//  // initializing parameters
  val (m,p) = (x.rows, x.cols)
  val theta = DenseVector.tabulate(nMixtures){i => 1/nMixtures.toDouble}

  val (means, z) = if(initParams == "random") {
    val random = Rand.randInt(0, x.rows)
    val temp = if(meansInit == null) DenseMatrix.zeros[Double](nMixtures, x.cols) else meansInit
    for(i <- 0 until x.cols) {
      temp(::, i) := temp(::, i).map{ elem => x(random.draw, i)}
    }
    (temp, DenseMatrix.ones[Double](x.rows, nMixtures) / nMixtures.toDouble)
  } else {
    val kmeans = new Kmeans(x, nMixtures)
    kmeans.fit()
    val temp = DenseMatrix.zeros[Double](x.rows, nMixtures)
    for(i <- 0 until x.rows) {
      temp(i, kmeans.clusterAssignment(i)) = 1d
    }
    (kmeans.centroids, temp)
  }
  // val z = DenseMatrix.ones[Double](x.rows, nMixtures) / nMixtures.toDouble
  val resp = DenseMatrix.ones[Double](x.rows, nMixtures)
  val covMat = fullCov(x, means, z, nMixtures, regCovariance)
  val dists = ( 0 until nMixtures) map { i =>
    MultivariateGaussian(means(i, ::).t, covMat(i))
  } toArray
  // end intialization
//
def fit(nIters: Int = 10) = {
for(ii <- 0 to nIters) {
  // e step
  for(i <- 0 until nMixtures) {
    resp(::, i) := x(*, ::).map{ e => dists(i).pdf(e)}
  }
  val respNormalizer = resp * theta
  for(i <- 0 until nMixtures) {
    z(::, i) := theta(i) * resp(::, i) /:/ respNormalizer
  }
  // m step
  means := fullMeans(x, z)
  covMat := fullCov(x, means, z, nMixtures, regCovariance)
  theta := mean(z(::, *)).t
  println( sum( log( resp * theta ))/x.rows.toDouble)
  for(i <- 0 until nMixtures) dists(i) = MultivariateGaussian(means(i, ::).t, covMat(i))
  }
}
//

  // covarianceType
  // "full" (each component has its own general covariance matrix),
  // "tied" (all components share the same general covariance matrix),
  // "diag" (each component has its own diagonal covariance matrix),
  // "spherical" (each component has its own single variance).

  //covarianceType match {
  //  "full" => fullCov
  //  "tied" => tiedCov
  //  "diag" => diagCov
  //  "spherical" => sphericalCov
  //}

  // compute cholesky
  // val chol = cholesky(x) =>
  // val invChol = inv(chol)
  // invChol.t * invChol = inv(x)
  //def fit(tolerance: double = 1e-6, maxIters: Int = 1000) = {
  //def goodEnough(isTrue: Boolean): Unit = {
//  if(isTrue) Unit
//  else {
//  }
//}
 //}
}
