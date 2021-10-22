package vct.main;

import hre.config.IntegerSetting;

import java.io.File;
import java.nio.file.Path;

import vct.col.ast.stmt.decl.ProgramUnit;
import vct.parsers.Parser;
import vct.parsers.ColCParser;
import vct.parsers.ColIParser;
import vct.parsers.ColJavaParser;
import vct.parsers.ColPVLParser;
import vct.silver.ColSilverParser;

import static hre.lang.System.*;

public class Parsers {
  public static Parser getParser(String extension){
    switch(extension){
      case "cl":
      case "c":
      case "cu":
        return new ColCParser();
      case "i":
        return new ColIParser();
      case "java7":
      case "java8":
      case "java":
        return new ColJavaParser(false);
      case "jspec":
        return new ColJavaParser(true);
      case "pvl":
        return new ColPVLParser();
      case "sil":
        return new ColSilverParser();
    }
    Fail("no parser for %s is known",extension);
    return null;
  }

  public static ProgramUnit.SourceLanguage getLanguage(String extension){
    switch(extension){
      case "cl":
      case "c":
        return ProgramUnit.SourceLanguage.OpenCLorC;
      case "cu":
        return ProgramUnit.SourceLanguage.CUDA;
      case "i":
        return ProgramUnit.SourceLanguage.I;
      case "java7":
      case "java8":
      case "java":
      case "jspec":
        return ProgramUnit.SourceLanguage.Java;
      case "pvl":
        return ProgramUnit.SourceLanguage.PVL;
      case "sil":
        return ProgramUnit.SourceLanguage.Silver;
    }
    Fail("no parser for %s is known",extension);
    return null;
  }
  
  public static ProgramUnit parseFile(Path filePath) {
    String name = filePath.toString();
    int dot = name.lastIndexOf('.');
    int p = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
    if (dot < 0 || dot < p) {
      Fail("cannot deduce language of %s", filePath);
    }
    String lang = name.substring(dot + 1);
    Progress("Parsing %s file %s", lang, filePath);
    Parser parser = Parsers.getParser(lang);
    if (parser == null) {
      Abort("Cannot detect language for extension \".%s\"", lang);
      return null;
    } else {
      ProgramUnit unit = Parsers.getParser(lang).parse(filePath.toFile());
      ProgramUnit.SourceLanguage sourceLanguage = getLanguage(lang);
      unit.sourceLanguage = sourceLanguage;
      Progress("Read %s successfully", name);
      return unit;
    }
  }

}
