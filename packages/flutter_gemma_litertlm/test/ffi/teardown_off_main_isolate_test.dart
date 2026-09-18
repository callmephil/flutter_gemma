import 'dart:async';
import 'dart:ffi';

import 'package:flutter_gemma_litertlm/src/ffi/litert_lm_bindings.dart';
import 'package:flutter_gemma_litertlm/src/ffi/litert_lm_client.dart';
import 'package:flutter_test/flutter_test.dart';

/// The two native calls that used to run on the main isolate, measured on an
/// S25 Ultra while Scan City asked its first question:
///
///   * the first `dlopen` of libLiteRtLm plus the default-scope probe, ~100ms,
///     before `initialize` even spawned its engine-create isolate;
///   * `litert_lm_conversation_delete`, 65-80ms, on the way into EVERY
///     `createChat` — a new chat closes the previous conversation first,
///     because the engine holds one at a time (upstream #966).
///
/// Both are now dispatched through [offMainIsolate] (`Isolate.run` in
/// production). These tests replace that seam, so what they can see is the
/// dispatch itself: the work never runs on the isolate the test is on, which
/// is exactly the claim — nothing native happens on the main isolate.
///
/// What they cannot see is the timing. That needs a phone.
void main() {
  late Future<void> Function(void Function()) originalRunner;
  late List<void Function()> dispatched;
  late List<Completer<void>> gates;

  setUp(() {
    originalRunner = offMainIsolate;
    dispatched = [];
    gates = [];
    // Hand back a gate instead of running the work: a unit test has no native
    // library behind it, and holding the dispatch open is what lets the
    // serialization and the shutdown drain be observed.
    offMainIsolate = (work) {
      dispatched.add(work);
      final gate = Completer<void>();
      gates.add(gate);
      return gate.future;
    };
    nativeLibrariesLoadedForTest = true;
  });

  tearDown(() {
    offMainIsolate = originalRunner;
    nativeLibrariesLoadedForTest = false;
  });

  Pointer<LiteRtLmConversation> conv(int address) =>
      Pointer<LiteRtLmConversation>.fromAddress(address);

  test('a conversation is freed on a spawned isolate, never inline', () async {
    final client = LiteRtLmFfiClient();
    final a = conv(0xC0FFEE);
    client.registerLiveForTest(a);

    final free = client.deleteConversationForTest(a);

    expect(
      client.isConversationLiveForTest(a),
      isFalse,
      reason:
          'liveness has to drop synchronously, before the free is dispatched: '
          'between here and the native delete a late onCancel must find the '
          'conversation dead rather than dereference it (#379)',
    );
    await pumpEventQueue();
    expect(
      dispatched,
      hasLength(1),
      reason: 'the native delete went to the isolate runner',
    );

    gates.single.complete();
    await free;
  });

  test('a second free waits for the first — the engine is not reentrant', () async {
    final client = LiteRtLmFfiClient();
    final a = conv(0xA000);
    final b = conv(0xB000);
    client.registerLiveForTest(a);
    client.registerLiveForTest(b);

    client.deleteConversationDeferredForTest(a);
    client.deleteConversationDeferredForTest(b);
    await pumpEventQueue();

    expect(
      dispatched,
      hasLength(1),
      reason:
          'the second free is queued on the native mutex; two isolates inside '
          'liblitert_lm on one engine at once is what the mutex exists to stop',
    );

    gates.first.complete();
    await pumpEventQueue();
    expect(dispatched, hasLength(2));

    gates.last.complete();
    await pumpEventQueue();
    expect(client.pendingDeleteCountForTest, 0);
  });

  test('shutdown waits for a free that is still on an isolate', () async {
    final client = LiteRtLmFfiClient();
    final a = conv(0xD000);
    client.registerLiveForTest(a);

    // The synchronous-by-contract path: ConversationHandle.close() cannot
    // return a future, so nobody but shutdown can wait for this one.
    client.deleteConversationDeferredForTest(a);
    await pumpEventQueue();
    expect(client.pendingDeleteCountForTest, 1);

    var shutdownDone = false;
    unawaited(client.shutdown().then((_) => shutdownDone = true));
    await pumpEventQueue();

    expect(
      shutdownDone,
      isFalse,
      reason:
          'engine_delete bulk-frees every conversation the engine owns, so '
          'running it while an isolate is inside conversation_delete would '
          'free the same object twice',
    );

    gates.single.complete();
    await pumpEventQueue();
    expect(shutdownDone, isTrue);
    expect(client.pendingDeleteCountForTest, 0);
  });

  test('a free dispatched before the libraries loaded stays in Dart', () async {
    // No library in the process means no library for the isolate to open by
    // name — and a client that never initialized has no conversation to free.
    nativeLibrariesLoadedForTest = false;
    final client = LiteRtLmFfiClient();
    final a = conv(0xE000);
    client.registerLiveForTest(a);

    await client.deleteConversationForTest(a);

    expect(dispatched, isEmpty);
    expect(client.isConversationLiveForTest(a), isFalse);
  });

  test('initialize loads the libraries on an isolate before the main one '
      'touches the loader', () async {
    nativeLibrariesLoadedForTest = false;
    var warmCalls = 0;
    offMainIsolate = (work) {
      warmCalls++;
      return Future<void>.error(
        UnsupportedError('no native library on the test host'),
      );
    };

    final client = LiteRtLmFfiClient();

    await expectLater(
      client.initialize(modelPath: '/no/such/model.litertlm'),
      throwsA(
        isA<UnsupportedError>().having(
          (e) => e.message,
          'message',
          'no native library on the test host',
        ),
      ),
      reason:
          'the warm failed, so initialize failed with ITS error — had the main '
          'isolate gone on to open the libraries itself, the failure would be '
          "the loader's instead, and that open is the ~100ms UI block",
    );

    expect(warmCalls, 1);
    expect(
      nativeLibrariesLoadedForTest,
      isFalse,
      reason: 'a load that threw is not a load: the next attempt retries it',
    );
  });
}
