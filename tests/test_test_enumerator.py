#!/usr/bin/env python3
"""Checagem do parser de saída do javap.

Cobre os quatro casos que faziam o enumerador antigo perder trigger tests, todos
tirados de saídas reais do Defects4J:

- método herdado de classe-base abstrata (Math-12, testDistributionClone);
- anotação @Test muito depois da assinatura, por bytecode longo (Mockito-1);
- anotação com argumentos, que o javap quebra em várias linhas
  (Compress-28, @Test(expected=IOException.class));
- nome de teste com maiúscula (Lang-1, TestLang747).

Uso: python3 tests/test_test_enumerator.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.test_enumerator import TestEnumerator


def enumerator() -> TestEnumerator:
    te = TestEnumerator.__new__(TestEnumerator)
    te.logger = logging.getLogger("test")
    te._class_cache = {}
    return te


def method_block(decl: str, body_lines: int, annotation: str = None) -> str:
    """Bloco de um método como o javap o imprime, com bytecode de tamanho dado."""
    lines = [f"  {decl};", "    descriptor: ()V", "    flags: (0x0001) ACC_PUBLIC", "    Code:"]
    lines += [f"        {i}: aload_0" for i in range(body_lines)]
    if annotation:
        lines += ["    RuntimeVisibleAnnotations:", "      0: #119()", f"        {annotation}"]
    return "\n".join(lines)


SUBCLASS = """Classfile /x/GammaDistributionTest.class
public class org.foo.GammaDistributionTest extends org.foo.RealDistributionAbstractTest
  minor version: 0
  major version: 55
  flags: (0x0021) ACC_PUBLIC, ACC_SUPER
  this_class: #298                        // org/foo/GammaDistributionTest
  super_class: #299                       // org/foo/RealDistributionAbstractTest
  interfaces: 0, fields: 1, methods: 3, attributes: 3
Constant pool:
  #1 = Utf8               whatever
{
""" + method_block("public void testDensity()", 3, "org.junit.Test") + "\n" \
  + method_block("public void helper(int)", 2) + "\n" \
  + method_block("public static void setUpClass()", 1, "org.junit.BeforeClass") + "\n}"

ABSTRACT_BASE = """Classfile /x/RealDistributionAbstractTest.class
public abstract class org.foo.RealDistributionAbstractTest
  minor version: 0
  major version: 55
  flags: (0x0421) ACC_PUBLIC, ACC_SUPER, ACC_ABSTRACT
  this_class: #108                        // org/foo/RealDistributionAbstractTest
  super_class: #109                       // java/lang/Object
  interfaces: 0, fields: 7, methods: 4, attributes: 4
Constant pool:
  #1 = Utf8               whatever
{
""" + method_block("public void testDistributionClone()", 300, "org.junit.Test") + "\n" \
  + method_block("public void shouldRunBecauseAnnotated()", 250, "org.junit.Test") + "\n" \
  + method_block("public void testSkippedBecauseIgnored()", 5, "org.junit.Ignore") + "\n}"

ANNOTATION_WITH_ARGS = """Classfile /x/TarArchiveInputStreamTest.class
public class org.foo.TarArchiveInputStreamTest
  minor version: 0
  major version: 50
  flags: (0x0021) ACC_PUBLIC, ACC_SUPER
  this_class: #77                         // org/foo/TarArchiveInputStreamTest
  super_class: #99                        // java/lang/Object
  interfaces: 0, fields: 0, methods: 2, attributes: 1
Constant pool:
  #1 = Utf8               whatever
{
  public void shouldThrowAnExceptionOnTruncatedEntries() throws java.lang.Exception;
    descriptor: ()V
    flags: (0x0001) ACC_PUBLIC
    Code:
         0: aload_0
    Exceptions:
      throws java.lang.Exception
    RuntimeVisibleAnnotations:
      0: #119(#158=c#142)
        org.junit.Test(
          expected=class Ljava/io/IOException;
        )
  public void TestLang747() throws java.lang.Exception;
    descriptor: ()V
    flags: (0x0001) ACC_PUBLIC
    Code:
         0: aload_0
}"""


def main() -> int:
    te = enumerator()

    # --- subclasse: superclasse detectada, construtor e helpers fora
    sub = te.parse_javap_output(SUBCLASS, "org.foo.GammaDistributionTest")
    assert sub.superclass == "org.foo.RealDistributionAbstractTest", sub.superclass
    assert sub.is_abstract is False
    tests = {m.name for m in sub.methods if m.is_test}
    assert tests == {"testDensity"}, tests
    # método com argumento não é teste
    assert not next(m for m in sub.methods if m.name == "helper").is_test
    # método estático anotado com @BeforeClass não é teste
    assert not next(m for m in sub.methods if m.name == "setUpClass").is_test

    # --- classe-base: abstrata, anotação a 300 linhas da assinatura, @Ignore fora
    base = te.parse_javap_output(ABSTRACT_BASE, "org.foo.RealDistributionAbstractTest")
    assert base.is_abstract is True
    assert base.superclass is None, base.superclass
    base_tests = {m.name for m in base.methods if m.is_test}
    assert "testDistributionClone" in base_tests, base_tests
    assert "shouldRunBecauseAnnotated" in base_tests, base_tests
    assert "testSkippedBecauseIgnored" not in base_tests, base_tests

    # --- herança: os testes da base contam pela subclasse
    te._class_cache = {
        ("org.foo.GammaDistributionTest", "/x"): sub,
        ("org.foo.RealDistributionAbstractTest", "/x"): base,
    }
    te.find_class_file = lambda name, bin_dir: Path("/x")
    inherited = {m.method_name for m in te.enumerate_class_methods("org.foo.GammaDistributionTest", Path("/x"))}
    assert inherited == {"testDensity", "testDistributionClone", "shouldRunBecauseAnnotated"}, inherited
    # todos atribuídos à subclasse, como o Defects4J nomeia o trigger
    assert all(
        m.class_name == "org.foo.GammaDistributionTest"
        for m in te.enumerate_class_methods("org.foo.GammaDistributionTest", Path("/x"))
    )

    # --- classe abstrata não é enumerada como executável
    assert te.enumerate_class_methods("org.foo.RealDistributionAbstractTest", Path("/x")) == []

    # --- anotação com argumentos e nome com maiúscula
    args = te.parse_javap_output(ANNOTATION_WITH_ARGS, "org.foo.TarArchiveInputStreamTest")
    arg_tests = {m.name for m in args.methods if m.is_test}
    assert "shouldThrowAnExceptionOnTruncatedEntries" in arg_tests, arg_tests
    assert "TestLang747" in arg_tests, arg_tests

    print("test_test_enumerator.py: todos os checks passaram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
