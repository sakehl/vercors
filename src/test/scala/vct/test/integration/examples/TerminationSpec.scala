package vct.test.integration.examples

import vct.test.integration.helper.VercorsSpec

class TerminationSpec extends VercorsSpec {
  vercors should verify using anyBackend example "concepts/decreases/integer.pvl"
}
