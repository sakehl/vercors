package vct.col.ast.statement.veymont

import vct.col.ast.{Communicate, Endpoint, EndpointName, Type}
import vct.col.check.{CheckContext, CheckError, SeqProgParticipant}
import vct.col.print.{Ctx, Doc, Group, Nest, Text}
import vct.col.ref.Ref
import vct.col.ast.ops.CommunicateOps
import vct.col.ast.ops.{CommunicateFamilyOps, CommunicateOps}

trait CommunicateImpl[G]
    extends CommunicateOps[G] with CommunicateFamilyOps[G] {
  comm: Communicate[G] =>
  override def layout(implicit ctx: Ctx): Doc =
    Text("channel_invariant") <+> Nest(invariant.show) <> ";" <+/> Group(
      Text("communicate") <+> layoutParticipant(receiver) <> target.show <+>
        "<-" <+> layoutParticipant(sender) <> msg.show <> ";"
    )

  def layoutParticipant(endpoint: Option[Ref[G, Endpoint[G]]])(
      implicit ctx: Ctx
  ) = endpoint.map(ref => Text(ctx.name(ref)) <> ": ").getOrElse(Text(""))

  override def check(context: CheckContext[G]): Seq[CheckError] =
    this match {
      case comm: Communicate[G]
          if sender.isDefined &&
            !context.currentParticipatingEndpoints.get
              .contains(sender.get.decl) =>
        Seq(SeqProgParticipant(sender.get.decl))
      case comm: Communicate[G]
          if receiver.isDefined &&
            !context.currentParticipatingEndpoints.get
              .contains(receiver.get.decl) =>
        Seq(SeqProgParticipant(receiver.get.decl))
      case _ => Nil
    }

  def participants: Seq[Endpoint[G]] = (sender.toSeq ++ receiver.toSeq)
    .map(_.decl)

  object t {
    def sender = comm.sender.get.decl.t
    def receiver = comm.receiver.get.decl.t
  }
}
