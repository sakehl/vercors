package vct.test.integration.examples

import vct.test.integration.helper.VercorsSpec

class BitVectorSpec extends VercorsSpec {
  vercors should verify using silicon example "examples/concepts/bitvectors/basic.c"
}
