// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: deephaven/proto/ticket.proto

#include "deephaven/proto/ticket.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

namespace io {
namespace deephaven {
namespace proto {
namespace backplane {
namespace grpc {
PROTOBUF_CONSTEXPR Ticket::Ticket(
    ::_pbi::ConstantInitialized)
  : ticket_(&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}){}
struct TicketDefaultTypeInternal {
  PROTOBUF_CONSTEXPR TicketDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~TicketDefaultTypeInternal() {}
  union {
    Ticket _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 TicketDefaultTypeInternal _Ticket_default_instance_;
PROTOBUF_CONSTEXPR TypedTicket::TypedTicket(
    ::_pbi::ConstantInitialized)
  : type_(&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{})
  , ticket_(nullptr){}
struct TypedTicketDefaultTypeInternal {
  PROTOBUF_CONSTEXPR TypedTicketDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~TypedTicketDefaultTypeInternal() {}
  union {
    TypedTicket _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 TypedTicketDefaultTypeInternal _TypedTicket_default_instance_;
}  // namespace grpc
}  // namespace backplane
}  // namespace proto
}  // namespace deephaven
}  // namespace io
static ::_pb::Metadata file_level_metadata_deephaven_2fproto_2fticket_2eproto[2];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_deephaven_2fproto_2fticket_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_deephaven_2fproto_2fticket_2eproto = nullptr;

const uint32_t TableStruct_deephaven_2fproto_2fticket_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::io::deephaven::proto::backplane::grpc::Ticket, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::io::deephaven::proto::backplane::grpc::Ticket, ticket_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::io::deephaven::proto::backplane::grpc::TypedTicket, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::io::deephaven::proto::backplane::grpc::TypedTicket, ticket_),
  PROTOBUF_FIELD_OFFSET(::io::deephaven::proto::backplane::grpc::TypedTicket, type_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::io::deephaven::proto::backplane::grpc::Ticket)},
  { 7, -1, -1, sizeof(::io::deephaven::proto::backplane::grpc::TypedTicket)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::io::deephaven::proto::backplane::grpc::_Ticket_default_instance_._instance,
  &::io::deephaven::proto::backplane::grpc::_TypedTicket_default_instance_._instance,
};

const char descriptor_table_protodef_deephaven_2fproto_2fticket_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\034deephaven/proto/ticket.proto\022!io.deeph"
  "aven.proto.backplane.grpc\"\030\n\006Ticket\022\016\n\006t"
  "icket\030\001 \001(\014\"V\n\013TypedTicket\0229\n\006ticket\030\001 \001"
  "(\0132).io.deephaven.proto.backplane.grpc.T"
  "icket\022\014\n\004type\030\002 \001(\tBBH\001P\001Z<github.com/de"
  "ephaven/deephaven-core/go/internal/proto"
  "/ticketb\006proto3"
  ;
static ::_pbi::once_flag descriptor_table_deephaven_2fproto_2fticket_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_deephaven_2fproto_2fticket_2eproto = {
    false, false, 255, descriptor_table_protodef_deephaven_2fproto_2fticket_2eproto,
    "deephaven/proto/ticket.proto",
    &descriptor_table_deephaven_2fproto_2fticket_2eproto_once, nullptr, 0, 2,
    schemas, file_default_instances, TableStruct_deephaven_2fproto_2fticket_2eproto::offsets,
    file_level_metadata_deephaven_2fproto_2fticket_2eproto, file_level_enum_descriptors_deephaven_2fproto_2fticket_2eproto,
    file_level_service_descriptors_deephaven_2fproto_2fticket_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_deephaven_2fproto_2fticket_2eproto_getter() {
  return &descriptor_table_deephaven_2fproto_2fticket_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_deephaven_2fproto_2fticket_2eproto(&descriptor_table_deephaven_2fproto_2fticket_2eproto);
namespace io {
namespace deephaven {
namespace proto {
namespace backplane {
namespace grpc {

// ===================================================================

class Ticket::_Internal {
 public:
};

Ticket::Ticket(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:io.deephaven.proto.backplane.grpc.Ticket)
}
Ticket::Ticket(const Ticket& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ticket_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    ticket_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_ticket().empty()) {
    ticket_.Set(from._internal_ticket(), 
      GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:io.deephaven.proto.backplane.grpc.Ticket)
}

inline void Ticket::SharedCtor() {
ticket_.InitDefault();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  ticket_.Set("", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

Ticket::~Ticket() {
  // @@protoc_insertion_point(destructor:io.deephaven.proto.backplane.grpc.Ticket)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void Ticket::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  ticket_.Destroy();
}

void Ticket::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void Ticket::Clear() {
// @@protoc_insertion_point(message_clear_start:io.deephaven.proto.backplane.grpc.Ticket)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  ticket_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Ticket::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // bytes ticket = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_ticket();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* Ticket::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:io.deephaven.proto.backplane.grpc.Ticket)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // bytes ticket = 1;
  if (!this->_internal_ticket().empty()) {
    target = stream->WriteBytesMaybeAliased(
        1, this->_internal_ticket(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:io.deephaven.proto.backplane.grpc.Ticket)
  return target;
}

size_t Ticket::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:io.deephaven.proto.backplane.grpc.Ticket)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // bytes ticket = 1;
  if (!this->_internal_ticket().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::BytesSize(
        this->_internal_ticket());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Ticket::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    Ticket::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Ticket::GetClassData() const { return &_class_data_; }

void Ticket::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<Ticket *>(to)->MergeFrom(
      static_cast<const Ticket &>(from));
}


void Ticket::MergeFrom(const Ticket& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:io.deephaven.proto.backplane.grpc.Ticket)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_ticket().empty()) {
    _internal_set_ticket(from._internal_ticket());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Ticket::CopyFrom(const Ticket& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:io.deephaven.proto.backplane.grpc.Ticket)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Ticket::IsInitialized() const {
  return true;
}

void Ticket::InternalSwap(Ticket* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &ticket_, lhs_arena,
      &other->ticket_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata Ticket::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_deephaven_2fproto_2fticket_2eproto_getter, &descriptor_table_deephaven_2fproto_2fticket_2eproto_once,
      file_level_metadata_deephaven_2fproto_2fticket_2eproto[0]);
}

// ===================================================================

class TypedTicket::_Internal {
 public:
  static const ::io::deephaven::proto::backplane::grpc::Ticket& ticket(const TypedTicket* msg);
};

const ::io::deephaven::proto::backplane::grpc::Ticket&
TypedTicket::_Internal::ticket(const TypedTicket* msg) {
  return *msg->ticket_;
}
TypedTicket::TypedTicket(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  // @@protoc_insertion_point(arena_constructor:io.deephaven.proto.backplane.grpc.TypedTicket)
}
TypedTicket::TypedTicket(const TypedTicket& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  type_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    type_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_type().empty()) {
    type_.Set(from._internal_type(), 
      GetArenaForAllocation());
  }
  if (from._internal_has_ticket()) {
    ticket_ = new ::io::deephaven::proto::backplane::grpc::Ticket(*from.ticket_);
  } else {
    ticket_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:io.deephaven.proto.backplane.grpc.TypedTicket)
}

inline void TypedTicket::SharedCtor() {
type_.InitDefault();
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  type_.Set("", GetArenaForAllocation());
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
ticket_ = nullptr;
}

TypedTicket::~TypedTicket() {
  // @@protoc_insertion_point(destructor:io.deephaven.proto.backplane.grpc.TypedTicket)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void TypedTicket::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  type_.Destroy();
  if (this != internal_default_instance()) delete ticket_;
}

void TypedTicket::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void TypedTicket::Clear() {
// @@protoc_insertion_point(message_clear_start:io.deephaven.proto.backplane.grpc.TypedTicket)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  type_.ClearToEmpty();
  if (GetArenaForAllocation() == nullptr && ticket_ != nullptr) {
    delete ticket_;
  }
  ticket_ = nullptr;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TypedTicket::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // .io.deephaven.proto.backplane.grpc.Ticket ticket = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr = ctx->ParseMessage(_internal_mutable_ticket(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string type = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          auto str = _internal_mutable_type();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "io.deephaven.proto.backplane.grpc.TypedTicket.type"));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* TypedTicket::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:io.deephaven.proto.backplane.grpc.TypedTicket)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // .io.deephaven.proto.backplane.grpc.Ticket ticket = 1;
  if (this->_internal_has_ticket()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, _Internal::ticket(this),
        _Internal::ticket(this).GetCachedSize(), target, stream);
  }

  // string type = 2;
  if (!this->_internal_type().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_type().data(), static_cast<int>(this->_internal_type().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "io.deephaven.proto.backplane.grpc.TypedTicket.type");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_type(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:io.deephaven.proto.backplane.grpc.TypedTicket)
  return target;
}

size_t TypedTicket::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:io.deephaven.proto.backplane.grpc.TypedTicket)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string type = 2;
  if (!this->_internal_type().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_type());
  }

  // .io.deephaven.proto.backplane.grpc.Ticket ticket = 1;
  if (this->_internal_has_ticket()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *ticket_);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData TypedTicket::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    TypedTicket::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*TypedTicket::GetClassData() const { return &_class_data_; }

void TypedTicket::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message* to,
                      const ::PROTOBUF_NAMESPACE_ID::Message& from) {
  static_cast<TypedTicket *>(to)->MergeFrom(
      static_cast<const TypedTicket &>(from));
}


void TypedTicket::MergeFrom(const TypedTicket& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:io.deephaven.proto.backplane.grpc.TypedTicket)
  GOOGLE_DCHECK_NE(&from, this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_type().empty()) {
    _internal_set_type(from._internal_type());
  }
  if (from._internal_has_ticket()) {
    _internal_mutable_ticket()->::io::deephaven::proto::backplane::grpc::Ticket::MergeFrom(from._internal_ticket());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void TypedTicket::CopyFrom(const TypedTicket& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:io.deephaven.proto.backplane.grpc.TypedTicket)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TypedTicket::IsInitialized() const {
  return true;
}

void TypedTicket::InternalSwap(TypedTicket* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &type_, lhs_arena,
      &other->type_, rhs_arena
  );
  swap(ticket_, other->ticket_);
}

::PROTOBUF_NAMESPACE_ID::Metadata TypedTicket::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_deephaven_2fproto_2fticket_2eproto_getter, &descriptor_table_deephaven_2fproto_2fticket_2eproto_once,
      file_level_metadata_deephaven_2fproto_2fticket_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace grpc
}  // namespace backplane
}  // namespace proto
}  // namespace deephaven
}  // namespace io
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::io::deephaven::proto::backplane::grpc::Ticket*
Arena::CreateMaybeMessage< ::io::deephaven::proto::backplane::grpc::Ticket >(Arena* arena) {
  return Arena::CreateMessageInternal< ::io::deephaven::proto::backplane::grpc::Ticket >(arena);
}
template<> PROTOBUF_NOINLINE ::io::deephaven::proto::backplane::grpc::TypedTicket*
Arena::CreateMaybeMessage< ::io::deephaven::proto::backplane::grpc::TypedTicket >(Arena* arena) {
  return Arena::CreateMessageInternal< ::io::deephaven::proto::backplane::grpc::TypedTicket >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>