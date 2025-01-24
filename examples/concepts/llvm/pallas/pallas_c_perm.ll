; ModuleID = './tmp/tmp_ir_source0.ll'
source_filename = "examples/concepts/llvm/pallas/pallas_c_perm.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i64, i64, i64, i64, i64, i64, i64 }
%pallas.fracT = type { i64, i64, i64, i64 }

@llvm.used = appending global [7 x ptr] [ptr @PALLAS_SPEC_0, ptr @PALLAS_SPEC_1, ptr @PALLAS_SPEC_2, ptr @PALLAS_SPEC_3, ptr @PALLAS_SPEC_4, ptr @PALLAS_SPEC_5, ptr @PALLAS_SPEC_6], section "llvm.metadata"

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @foo(ptr noundef %0) #0 !dbg !14 !pallas.fcontract !20 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !24, metadata !DIExpression()), !dbg !27
  %3 = load ptr, ptr %2, align 8, !dbg !28
  %4 = load i32, ptr %3, align 4, !dbg !29
  %5 = add nsw i32 %4, 5, !dbg !30
  ret i32 %5, !dbg !31
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind uwtable
define dso_local void @bar(ptr noundef %0) #0 !dbg !32 !pallas.fcontract !51 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  call void @llvm.dbg.declare(metadata ptr %2, metadata !55, metadata !DIExpression()), !dbg !64
  %3 = load ptr, ptr %2, align 8, !dbg !65
  %4 = getelementptr inbounds %struct.S, ptr %3, i32 0, i32 0, !dbg !66
  store i64 0, ptr %4, align 8, !dbg !67
  ret void, !dbg !68
}

; Function Attrs: noinline nounwind uwtable
define dso_local zeroext i1 @PALLAS_SPEC_0(ptr noundef %0) #0 !dbg !69 !pallas.exprWrapper !73 {
  %2 = alloca %pallas.fracT, align 8
  call void @llvm.dbg.value(metadata ptr %0, metadata !74, metadata !DIExpression()), !dbg !75
  %3 = icmp ne ptr %0, null, !dbg !76
  br i1 %3, label %4, label %6, !dbg !77

4:                                                ; preds = %1
  call void @pallas.fracOf(ptr sret(%pallas.fracT) %2, i32 noundef 1, i32 noundef 2), !dbg !78
  %5 = call i1 @pallas.perm(ptr noundef %0, ptr noundef byval(%pallas.fracT) %2), !dbg !79
  br label %6

6:                                                ; preds = %4, %1
  %7 = phi i1 [ false, %1 ], [ %5, %4 ], !dbg !75
  ret i1 %7, !dbg !75
}

; Function Attrs: noinline nounwind uwtable
define dso_local zeroext i1 @PALLAS_SPEC_1(ptr noundef %0) #0 !dbg !80 !pallas.exprWrapper !73 {
  %2 = alloca %pallas.fracT, align 8
  call void @llvm.dbg.value(metadata ptr %0, metadata !81, metadata !DIExpression()), !dbg !82
  call void @pallas.fracOf(ptr sret(%pallas.fracT) %2, i32 noundef 1, i32 noundef 2), !dbg !83
  %3 = call i1 @pallas.perm(ptr noundef %0, ptr noundef byval(%pallas.fracT) %2), !dbg !84
  ret i1 %3, !dbg !82
}

; Function Attrs: noinline nounwind uwtable
define dso_local zeroext i1 @PALLAS_SPEC_2(ptr noundef %0) #0 !dbg !85 !pallas.exprWrapper !73 {
  %2 = alloca %pallas.fracT, align 8
  call void @llvm.dbg.value(metadata ptr %0, metadata !100, metadata !DIExpression()), !dbg !101
  %3 = icmp ne ptr %0, null, !dbg !102
  br i1 %3, label %4, label %6, !dbg !103

4:                                                ; preds = %1
  call void @pallas.fracOf(ptr sret(%pallas.fracT) %2, i32 noundef 1, i32 noundef 2), !dbg !104
  %5 = call i1 @pallas.perm(ptr noundef %0, ptr noundef byval(%pallas.fracT) %2), !dbg !105
  br label %6

6:                                                ; preds = %4, %1
  %7 = phi i1 [ false, %1 ], [ %5, %4 ], !dbg !101
  ret i1 %7, !dbg !101
}

; Function Attrs: noinline nounwind uwtable
define dso_local zeroext i1 @PALLAS_SPEC_3(ptr noundef %0) #0 !dbg !106 !pallas.exprWrapper !73 {
  %2 = alloca %pallas.fracT, align 8
  call void @llvm.dbg.value(metadata ptr %0, metadata !107, metadata !DIExpression()), !dbg !108
  %3 = getelementptr inbounds %struct.S, ptr %0, i32 0, i32 0, !dbg !109
  call void @pallas.fracOf(ptr sret(%pallas.fracT) %2, i32 noundef 1, i32 noundef 1), !dbg !110
  %4 = call i1 @pallas.perm(ptr noundef %3, ptr noundef byval(%pallas.fracT) %2), !dbg !111
  ret i1 %4, !dbg !108
}

; Function Attrs: noinline nounwind uwtable
define dso_local zeroext i1 @PALLAS_SPEC_4(ptr noundef %0) #0 !dbg !112 !pallas.exprWrapper !73 {
  %2 = alloca %pallas.fracT, align 8
  call void @llvm.dbg.value(metadata ptr %0, metadata !113, metadata !DIExpression()), !dbg !114
  call void @pallas.fracOf(ptr sret(%pallas.fracT) %2, i32 noundef 1, i32 noundef 2), !dbg !115
  %3 = call i1 @pallas.perm(ptr noundef %0, ptr noundef byval(%pallas.fracT) %2), !dbg !116
  ret i1 %3, !dbg !114
}

; Function Attrs: noinline nounwind uwtable
define dso_local zeroext i1 @PALLAS_SPEC_5(ptr noundef %0) #0 !dbg !117 !pallas.exprWrapper !73 {
  %2 = alloca %pallas.fracT, align 8
  call void @llvm.dbg.value(metadata ptr %0, metadata !118, metadata !DIExpression()), !dbg !119
  %3 = getelementptr inbounds %struct.S, ptr %0, i32 0, i32 0, !dbg !120
  call void @pallas.fracOf(ptr sret(%pallas.fracT) %2, i32 noundef 1, i32 noundef 1), !dbg !121
  %4 = call i1 @pallas.perm(ptr noundef %3, ptr noundef byval(%pallas.fracT) %2), !dbg !122
  ret i1 %4, !dbg !119
}

; Function Attrs: noinline nounwind uwtable
define dso_local zeroext i1 @PALLAS_SPEC_6(ptr noundef %0) #0 !dbg !123 !pallas.exprWrapper !73 {
  call void @llvm.dbg.value(metadata ptr %0, metadata !124, metadata !DIExpression()), !dbg !125
  %2 = getelementptr inbounds %struct.S, ptr %0, i32 0, i32 0, !dbg !126
  %3 = load i64, ptr %2, align 8, !dbg !126
  %4 = icmp eq i64 %3, 0, !dbg !127
  ret i1 %4, !dbg !125
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

declare !pallas.specLib !128 i1 @pallas.perm(ptr noundef, ptr noundef byval(%pallas.fracT))

declare !pallas.specLib !129 void @pallas.fracOf(ptr sret(%pallas.fracT), i32 noundef, i32 noundef)

attributes #0 = { noinline nounwind uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0, !2}
!llvm.module.flags = !{!6, !7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (https://github.com/swiftlang/llvm-project.git 73500bf55acff5fa97b56dcdeb013f288efd084f)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "examples/concepts/llvm/pallas/pallas_c_perm.c", directory: ".", checksumkind: CSK_MD5, checksum: "e5185f5b1fb142a80500224dcb028130")
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (https://github.com/swiftlang/llvm-project.git 73500bf55acff5fa97b56dcdeb013f288efd084f)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "tmp/source_wrappers.c", directory: ".", checksumkind: CSK_MD5, checksum: "46ed619c671e33c5237dc7051723c00e")
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"PIE Level", i32 2}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{i32 7, !"frame-pointer", i32 2}
!13 = !{!"clang version 17.0.0 (https://github.com/swiftlang/llvm-project.git 73500bf55acff5fa97b56dcdeb013f288efd084f)"}
!14 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 8, type: !15, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !18}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!19 = !{}
!20 = !{!21, i1 false, !22, !25}
!21 = !{!"pallas.srcLoc", i64 4, i64 1, i64 7, i64 1}
!22 = !{!"pallas.requires", !23, ptr @PALLAS_SPEC_0, !24}
!23 = !{!"pallas.srcLoc", i64 5, i64 1, i64 5, i64 48}
!24 = !DILocalVariable(name: "ptr", arg: 1, scope: !14, file: !1, line: 8, type: !18)
!25 = !{!"pallas.ensures", !26, ptr @PALLAS_SPEC_1, !24}
!26 = !{!"pallas.srcLoc", i64 6, i64 1, i64 6, i64 32}
!27 = !DILocation(line: 8, column: 14, scope: !14)
!28 = !DILocation(line: 9, column: 13, scope: !14)
!29 = !DILocation(line: 9, column: 12, scope: !14)
!30 = !DILocation(line: 9, column: 17, scope: !14)
!31 = !DILocation(line: 9, column: 5, scope: !14)
!32 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 23, type: !33, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!33 = !DISubroutineType(types: !34)
!34 = !{null, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !36, size: 64)
!36 = !DIDerivedType(tag: DW_TAG_typedef, name: "BigStruct", file: !1, line: 14, baseType: !37)
!37 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, line: 12, size: 448, elements: !38)
!38 = !{!39, !45, !46, !47, !48, !49, !50}
!39 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !37, file: !1, line: 13, baseType: !40, size: 64)
!40 = !DIDerivedType(tag: DW_TAG_typedef, name: "int64_t", file: !41, line: 27, baseType: !42)
!41 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/stdint-intn.h", directory: "", checksumkind: CSK_MD5, checksum: "649b383a60bfa3eb90e85840b2b0be20")
!42 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int64_t", file: !43, line: 44, baseType: !44)
!43 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "", checksumkind: CSK_MD5, checksum: "e1865d9fe29fe1b5ced550b7ba458f9e")
!44 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !37, file: !1, line: 13, baseType: !40, size: 64, offset: 64)
!46 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !37, file: !1, line: 13, baseType: !40, size: 64, offset: 128)
!47 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !37, file: !1, line: 13, baseType: !40, size: 64, offset: 192)
!48 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !37, file: !1, line: 13, baseType: !40, size: 64, offset: 256)
!49 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !37, file: !1, line: 13, baseType: !40, size: 64, offset: 320)
!50 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !37, file: !1, line: 13, baseType: !40, size: 64, offset: 384)
!51 = !{!52, i1 false, !53, !56, !58, !60, !62}
!52 = !{!"pallas.srcLoc", i64 16, i64 1, i64 22, i64 1}
!53 = !{!"pallas.requires", !54, ptr @PALLAS_SPEC_2, !55}
!54 = !{!"pallas.srcLoc", i64 17, i64 1, i64 17, i64 44}
!55 = !DILocalVariable(name: "s", arg: 1, scope: !32, file: !1, line: 23, type: !35)
!56 = !{!"pallas.requires", !57, ptr @PALLAS_SPEC_3, !55}
!57 = !{!"pallas.srcLoc", i64 18, i64 1, i64 18, i64 35}
!58 = !{!"pallas.ensures", !59, ptr @PALLAS_SPEC_4, !55}
!59 = !{!"pallas.srcLoc", i64 19, i64 1, i64 19, i64 30}
!60 = !{!"pallas.ensures", !61, ptr @PALLAS_SPEC_5, !55}
!61 = !{!"pallas.srcLoc", i64 20, i64 1, i64 20, i64 34}
!62 = !{!"pallas.ensures", !63, ptr @PALLAS_SPEC_6, !55}
!63 = !{!"pallas.srcLoc", i64 21, i64 1, i64 21, i64 18}
!64 = !DILocation(line: 23, column: 21, scope: !32)
!65 = !DILocation(line: 24, column: 5, scope: !32)
!66 = !DILocation(line: 24, column: 8, scope: !32)
!67 = !DILocation(line: 24, column: 10, scope: !32)
!68 = !DILocation(line: 25, column: 1, scope: !32)
!69 = distinct !DISubprogram(name: "PALLAS_SPEC_0", scope: !1, file: !1, line: 5, type: !70, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!70 = !DISubroutineType(types: !71)
!71 = !{!72, !18}
!72 = !DIBasicType(name: "_Bool", size: 8, encoding: DW_ATE_boolean)
!73 = !{!""}
!74 = !DILocalVariable(name: "ptr", arg: 1, scope: !69, file: !1, line: 5, type: !18)
!75 = !DILocation(line: 0, scope: !69)
!76 = !DILocation(line: 5, column: 14, scope: !69)
!77 = !DILocation(line: 5, column: 22, scope: !69)
!78 = !DILocation(line: 5, column: 35, scope: !69)
!79 = !DILocation(line: 5, column: 25, scope: !69)
!80 = distinct !DISubprogram(name: "PALLAS_SPEC_1", scope: !1, file: !1, line: 6, type: !70, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!81 = !DILocalVariable(name: "ptr", arg: 1, scope: !80, file: !1, line: 6, type: !18)
!82 = !DILocation(line: 0, scope: !80)
!83 = !DILocation(line: 6, column: 19, scope: !80)
!84 = !DILocation(line: 6, column: 9, scope: !80)
!85 = distinct !DISubprogram(name: "PALLAS_SPEC_2", scope: !1, file: !1, line: 17, type: !86, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!86 = !DISubroutineType(types: !87)
!87 = !{!72, !88}
!88 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !89, size: 64)
!89 = !DIDerivedType(tag: DW_TAG_typedef, name: "BigStruct", file: !90, line: 29, baseType: !91)
!90 = !DIFile(filename: "./tmp/source_wrappers.c", directory: ".", checksumkind: CSK_MD5, checksum: "46ed619c671e33c5237dc7051723c00e")
!91 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !90, line: 27, size: 448, elements: !92)
!92 = !{!93, !94, !95, !96, !97, !98, !99}
!93 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !91, file: !90, line: 28, baseType: !40, size: 64)
!94 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !91, file: !90, line: 28, baseType: !40, size: 64, offset: 64)
!95 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !91, file: !90, line: 28, baseType: !40, size: 64, offset: 128)
!96 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !91, file: !90, line: 28, baseType: !40, size: 64, offset: 192)
!97 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !91, file: !90, line: 28, baseType: !40, size: 64, offset: 256)
!98 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !91, file: !90, line: 28, baseType: !40, size: 64, offset: 320)
!99 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !91, file: !90, line: 28, baseType: !40, size: 64, offset: 384)
!100 = !DILocalVariable(name: "s", arg: 1, scope: !85, file: !1, line: 17, type: !88)
!101 = !DILocation(line: 0, scope: !85)
!102 = !DILocation(line: 17, column: 12, scope: !85)
!103 = !DILocation(line: 17, column: 20, scope: !85)
!104 = !DILocation(line: 17, column: 31, scope: !85)
!105 = !DILocation(line: 17, column: 23, scope: !85)
!106 = distinct !DISubprogram(name: "PALLAS_SPEC_3", scope: !1, file: !1, line: 18, type: !86, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!107 = !DILocalVariable(name: "s", arg: 1, scope: !106, file: !1, line: 18, type: !88)
!108 = !DILocation(line: 0, scope: !106)
!109 = !DILocation(line: 18, column: 19, scope: !106)
!110 = !DILocation(line: 18, column: 22, scope: !106)
!111 = !DILocation(line: 18, column: 10, scope: !106)
!112 = distinct !DISubprogram(name: "PALLAS_SPEC_4", scope: !1, file: !1, line: 19, type: !86, scopeLine: 19, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!113 = !DILocalVariable(name: "s", arg: 1, scope: !112, file: !1, line: 19, type: !88)
!114 = !DILocation(line: 0, scope: !112)
!115 = !DILocation(line: 19, column: 17, scope: !112)
!116 = !DILocation(line: 19, column: 9, scope: !112)
!117 = distinct !DISubprogram(name: "PALLAS_SPEC_5", scope: !1, file: !1, line: 20, type: !86, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!118 = !DILocalVariable(name: "s", arg: 1, scope: !117, file: !1, line: 20, type: !88)
!119 = !DILocation(line: 0, scope: !117)
!120 = !DILocation(line: 20, column: 18, scope: !117)
!121 = !DILocation(line: 20, column: 21, scope: !117)
!122 = !DILocation(line: 20, column: 9, scope: !117)
!123 = distinct !DISubprogram(name: "PALLAS_SPEC_6", scope: !1, file: !1, line: 21, type: !86, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
!124 = !DILocalVariable(name: "s", arg: 1, scope: !123, file: !1, line: 21, type: !88)
!125 = !DILocation(line: 0, scope: !123)
!126 = !DILocation(line: 21, column: 12, scope: !123)
!127 = !DILocation(line: 21, column: 14, scope: !123)
!128 = !{!"pallas.perm"}
!129 = !{!"pallas.fracOf"}
