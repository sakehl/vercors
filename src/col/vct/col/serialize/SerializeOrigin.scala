package vct.col.serialize

import hre.io.{ChecksumReadableFile, RWFile}
import vct.col.ast.{serialize => ser}
import vct.col.origin._

import java.nio.file.Path
import scala.annotation.unused
import scala.collection.mutable
import com.typesafe.scalalogging.LazyLogging;

object SerializeOrigin extends LazyLogging {
  private val fileMap: mutable.HashMap[Path, hre.io.Readable] = mutable
    .HashMap()

  def deserialize(
      @unused
      origin: ser.Origin
  ): Origin =
    Origin(origin.content.map(_.content).flatMap {
      case ser.OriginContent.Content.SourceName(name) =>
        Seq(SourceName(name.name))
      case ser.OriginContent.Content.PreferredName(name) =>
        Seq(PreferredName(name.preferredName))
      case ser.OriginContent.Content.Context(context) =>
        Seq(DeserializedContext(
          context.context,
          context.inlineContext,
          context.shortPosition,
        ))
      case ser.OriginContent.Content.ReadableOrigin(context) =>
        val path = Path.of(context.directory, context.filename)
        val readable = fileMap.getOrElseUpdate(
          path, {
            if (context.filename == "<stdin>") {
              logger.warn(
                "The file was compiled from standard input, origin information will be missing"
              )
              null
            } else if (path.toFile.exists()) {
              if (
                context.checksum.isDefined && context.checksumKind.isDefined
              ) {
                val file = ChecksumReadableFile(
                  path,
                  doWatch = false,
                  context.checksumKind.get,
                )
                // TODO: Should we do the checksum check later? Potentially this causes a lot of file loading
                if (file.getChecksum != context.checksum.get) {
                  logger.warn(
                    s"The checksum of the file '$path' does not match the LLVM checksum error locations are likely inaccurate"
                  )
                }
                file
              } else { RWFile(path) }
            } else {
              logger.error(
                f"File not found: '$path', origin information will be missing"
              )
              null
            }
          },
        )
        if (readable == null) { Nil }
        else { Seq(ReadableOrigin(readable)) }
      case ser.OriginContent.Content.PositionRange(range) =>
        // TODO: Preserve the start col idx even if end col idx is missing and improve the origins in LangLLVMToCol? Maybe we could even set the preferred name correctly
        Seq(PositionRange(
          range.startLineIdx,
          range.endLineIdx,
          range.startColIdx.flatMap { start => range.endColIdx.map((start, _)) },
        ))
      case ser.OriginContent.Content.LabelContext(label) =>
        Seq(LabelContext(label.label))
    })

  def serialize(
      @unused
      origin: Origin
  ): ser.Origin =
    ser.Origin(
      origin.originContents.flatMap {
        case SourceName(name) =>
          Seq(ser.OriginContent.Content.SourceName(ser.SourceName(name)))
        case PreferredName(preferredName) =>
          Seq(
            ser.OriginContent.Content
              .PreferredName(ser.PreferredName(preferredName))
          )
        case DeserializedContext(context, inlineContext, shortPosition) =>
          Seq(
            ser.OriginContent.Content
              .Context(ser.Context(context, inlineContext, shortPosition))
          )
        case ReadableOrigin(readable) =>
          // Not sure how to best deal with directory/filename here
          Seq(ser.OriginContent.Content.ReadableOrigin(ser.ReadableOrigin(
            "",
            readable.underlyingPath.map(_.toString)
              .getOrElse(readable.fileName),
            None,
            None,
          )))
        case PositionRange(startLineIdx, endLineIdx, startEndColIdx) =>
          Seq(ser.OriginContent.Content.PositionRange(ser.PositionRange(
            startLineIdx,
            endLineIdx,
            startEndColIdx.map(_._1),
            startEndColIdx.map(_._2),
          )))
        case LabelContext(label) =>
          Seq(ser.OriginContent.Content.LabelContext(ser.LabelContext(label)))
        case _ => Nil
      }.map(ser.OriginContent(_))
    )
}

case class DeserializedContext(
    context: String,
    inlineContext: String,
    shortPosition: String,
) extends Context {
  override protected def contextHere(tail: Origin): (String, Origin) =
    (context, tail)
  override protected def inlineContextHere(
      tail: Origin,
      compress: Boolean = true,
  ): (String, Origin) = (inlineContext, tail)
  override protected def shortPositionHere(tail: Origin): (String, Origin) =
    (shortPosition, tail)
}
