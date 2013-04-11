package semper.carbon.modules

import semper.sil.{ast => sil}
import semper.carbon.boogie.{LocalVar, Exp}

/**
 * A module for translating SIL expressions.
 *
 * @author Stefan Heule
 */
trait ExpModule extends Module {
  def translateExp(exp: sil.Exp): Exp
  def translateLocalVar(l: sil.LocalVar): LocalVar
}
