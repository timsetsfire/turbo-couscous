import breeze.stats.distributions.{Uniform, RandBasis}
import breeze.linalg._
import breeze.numerics._
import java.util.Calendar

class Sampler(x: DenseVector[Double]) {
// if( sum(p) != 1d) throw new Exception("probabilites don't sum to 1")
	val initProbs = DenseVector.ones[Double](x.length) / x.length.toDouble
	def sampleWithReplacement(samples: Int, probs: DenseVector[Double] = this.initProbs, seed: Int = Calendar.getInstance.hashCode) = {
		 val cumlsum = accumulate(probs)
		 val s = DenseVector.zeros[Double](samples)
		 val u = Uniform(0,1)(RandBasis.withSeed(seed))
			s.map {
			elem =>
			val unif = u.draw
			val all = cumlsum.findAll(e => unif <= e)
			val index = min( all )
			x(index)
		}
	}
}
