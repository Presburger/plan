#include "Plan.pb.h"
#include "PlanBaseVisitor.h"
#include "PlanLexer.h"
#include "PlanParser.h"
#include <any>
#include <iostream>
#include <string>

namespace milvus {

struct ExprWithDtype {
  proto::plan::Expr *expr;
  proto::schema::DataType dtype;
  bool dependent;
  ExprWithDtype(proto::plan::Expr *const expr, proto::schema::DataType dtype,
                bool dependent)
      : expr(expr), dtype(dtype), dependent(dependent) {}
};

template <typename T> std::optional<T> extractValue(proto::plan::Expr *expr) {
  if (!expr->has_value_expr())
    return std::nullopt;

  if constexpr (std::is_same_v<T, std::int64_t>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_int64_val())
      return expr->value_expr().value().int64_val();
  }
  if constexpr (std::is_same_v<T, double>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_float_val())
      return expr->value_expr().value().float_val();
  }

  if constexpr (std::is_same_v<T, std::string>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_string_val())
      return expr->value_expr().value().string_val();
  }
  if constexpr (std::is_same_v<T, bool>) {
    if (expr->value_expr().has_value() &&
        expr->value_expr().value().has_bool_val())
      return expr->value_expr().value().bool_val();
  }

  return std::nullopt;
};

template <typename T> proto::plan::Expr *createValueExpr(const T val) {
  auto expr = new proto::plan::Expr();
  auto val_expr = new proto::plan::ValueExpr();

  auto value = new proto::plan::GenericValue();
  if constexpr (std::is_same_v<T, int64_t>)
    value->set_int64_val(val);
  else if constexpr (std::is_same_v<T, std::string>)
    value->set_string_val(val);
  else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>)
    value->set_float_val(val);
  else if constexpr (std::is_same_v<T, bool>)
    value->set_bool_val(val);
  else
    static_assert(false);

  val_expr->set_allocated_value(value);
  expr->set_allocated_value_expr(val_expr);
  return expr;
}

template <proto::plan::BinaryExpr_BinaryOp T>
proto::plan::Expr *createBinExpr(proto::plan::Expr *left,
                                 proto::plan::Expr *right) {

  auto expr = new proto::plan::Expr();
  auto bin_expr = new proto::plan::BinaryExpr();
  bin_expr->set_op(T);
  bin_expr->set_allocated_left(left);
  bin_expr->set_allocated_right(right);
  return expr;
}

template <proto::plan::ArithOpType T>
proto::plan::Expr *createBinArithExpr(proto::plan::Expr *left,
                                      proto::plan::Expr *right) {

  auto expr = new proto::plan::Expr();
  auto bin_expr = new proto::plan::BinaryArithExpr();
  bin_expr->set_op(T);
  bin_expr->set_allocated_left(left);
  bin_expr->set_allocated_right(right);
  return expr;
}

bool arithmeticDtype(proto::schema::DataType type) {
  switch (type) {
  case proto::schema::DataType::Float:
    return true;
  case proto::schema::DataType::Double:
    return true;
  case proto::schema::DataType::Int8:
    return true;
  case proto::schema::DataType::Int16:
    return true;
  case proto::schema::DataType::Int32:
    return true;
  case proto::schema::DataType::Int64:
    return true;
  default:
    return false;
  }
}

proto::schema::DataType getArrayElementType(proto::plan::Expr *expr) {
  if (expr->has_column_expr()) {
    return expr->column_expr().info().data_type();
  }
  if (expr->has_value_expr() && expr->value_expr().has_value() &&
      expr->value_expr().value().has_array_val()) {
    return expr->value_expr().value().array_val().element_type();
  }

  return proto::schema::DataType::None;
}

bool canArithmeticDtype(proto::schema::DataType left_type,
                        proto::schema::DataType right_type) {

  if (left_type == proto::schema::DataType::JSON &&
      right_type == proto::schema::DataType::JSON)
    return false;
  if (left_type == proto::schema::DataType::JSON && arithmeticDtype(right_type))
    return true;
  if (arithmeticDtype(left_type) && right_type == proto::schema::DataType::JSON)
    return true;
  if (arithmeticDtype(left_type) && arithmeticDtype(right_type))
    return true;
  return false;
}

proto::schema::DataType calDataType(ExprWithDtype *a, ExprWithDtype *b) {
  auto a_dtype = a->dtype;
  auto b_dtype = b->dtype;
  if (a->dtype == proto::schema::DataType::Array) {
    a_dtype = getArrayElementType(a->expr);
  }
  if (b->dtype == proto::schema::DataType::Array) {
    b_dtype = getArrayElementType(b->expr);
  }
  if (a_dtype == proto::schema::DataType::JSON) {
    if (b_dtype == proto::schema::DataType::Float ||
        b_dtype == proto::schema::DataType::Double)
      return proto::schema::DataType::Float;
    if (b_dtype == proto::schema::DataType::Int8 ||
        b_dtype == proto::schema::DataType::Int16 ||
        b_dtype == proto::schema::DataType::Int32 ||
        b_dtype == proto::schema::DataType::Int64)
      return proto::schema::DataType::Int64;
    if (b_dtype == proto::schema::DataType::JSON)
      return proto::schema::DataType::JSON;
  }

  if (a_dtype == proto::schema::DataType::Float ||
      a_dtype == proto::schema::DataType::Double) {
    if (b_dtype == proto::schema::DataType::JSON)
      return proto::schema::DataType::Double;
    if (arithmeticDtype(b_dtype))
      return proto::schema::DataType::Double;
  }

  if (a_dtype == proto::schema::DataType::Int8 ||
      a_dtype == proto::schema::DataType::Int16 ||
      a_dtype == proto::schema::DataType::Int32 ||
      a_dtype == proto::schema::DataType::Int64) {
    if (b_dtype == proto::schema::DataType::Float ||
        b_dtype == proto::schema::DataType::Double)
      return proto::schema::DataType::Double;
    if (b_dtype == proto::schema::DataType::Int8 ||
        b_dtype == proto::schema::DataType::Int16 ||
        b_dtype == proto::schema::DataType::Int32 ||
        b_dtype == proto::schema::DataType::Int64 ||
        b_dtype == proto::schema::DataType::JSON)
      return proto::schema::DataType::Int64;
  }
  assert(false);
}

struct SchemaHelper {

  SchemaHelper() = default;

  proto::schema::CollectionSchema *schema = nullptr;
  std::map<std::string, int> name_offset;
  std::map<int64_t, int> id_offset;

  int primary_key_offset = -1;
  int partition_key_offset = -1;

  const proto::schema::FieldSchema &GetPrimaryKeyField() {
    assert(primary_key_offset != -1);
    return schema->fields(primary_key_offset);
  }

  const proto::schema::FieldSchema &GetPartitionKeyField() {
    assert(partition_key_offset != -1);
    return schema->fields(partition_key_offset);
  }

  const proto::schema::FieldSchema &GetFieldFromName(const std::string &name) {
    auto it = name_offset.find(name);
    assert(it != name_offset.end());
    return schema->fields(it->second);
  }

  const proto::schema::FieldSchema &
  GetFieldFromNameDefaultJSON(const std::string &name) {
    auto it = name_offset.find(name);
    if (it == name_offset.end()) {
      return GetDefaultJSONField();
    }
    return schema->fields(it->second);
  }

  const proto::schema::FieldSchema &GetDefaultJSONField() {
    for (int i = 0; i < schema->fields_size(); ++i) {
      auto &field = schema->fields(i);
      if (field.data_type() == proto::schema::DataType::JSON &&
          field.is_dynamic())
        return field;
    }

    assert(false);
  }

  const proto::schema::FieldSchema &GetFieldFromID(int64_t id) {
    auto it = id_offset.find(id);
    assert(it != id_offset.end());
    return schema->fields(it->second);
  }

  int GetVectorDimFromID(int64_t id) {
    auto &field = GetFieldFromID(id);
    if (field.data_type() != proto::schema::DataType::FloatVector &&
        field.data_type() != proto::schema::DataType::Float16Vector &&
        field.data_type() != proto::schema::DataType::BinaryVector &&
        field.data_type() != proto::schema::DataType::BFloat16Vector) {
      assert(false);
    }
    for (int i = 0; i < field.type_params_size(); ++i) {
      if (field.type_params(i).key() == "dim")
        return std::stoi(field.type_params(i).value().c_str(), NULL, 10);
    }
    assert(false);
  }
};

SchemaHelper CreateSchemaHelper(proto::schema::CollectionSchema *schema) {

  assert(schema);
  SchemaHelper schema_helper;
  schema_helper.schema = schema;
  for (int i = 0; i < schema->fields_size(); ++i) {
    auto field = schema->fields(i);
    auto it = schema_helper.name_offset.find(field.name());
    if (it != schema_helper.name_offset.end())
      assert(false);
    schema_helper.name_offset[field.name()] = i;
    schema_helper.id_offset[field.fieldid()] = i;
    if (field.is_primary_key()) {
      assert(schema_helper.primary_key_offset != -1);
      schema_helper.primary_key_offset = i;
    }
    if (field.is_partition_key()) {
      assert(schema_helper.primary_key_offset != -1);
      schema_helper.partition_key_offset = i;
    }
  }
  return schema_helper;
}

std::string convertEscapeSingle(const std::string &str) { return str; }

std::pair<proto::plan::OpType, std::string>
translatePatternMatch(const std::string &pattern) {

  return std::make_pair(proto::plan::OpType::PrefixMatch, pattern);
}

bool canBeComparedDataType(proto::schema::DataType a,
                           proto::schema::DataType b) {
  switch (a) {
  case proto::schema::DataType::Bool:
    return (b == proto::schema::DataType::Bool) ||
           (b == proto::schema::DataType::JSON);
  case proto::schema::DataType::Int8:
    return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
  case proto::schema::DataType::Int16:
    return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
  case proto::schema::DataType::Int32:
    return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
  case proto::schema::DataType::Int64:
    return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
  case proto::schema::DataType::Float:
    return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
  case proto::schema::DataType::Double:
    return arithmeticDtype(b) || (b == proto::schema::DataType::JSON);
  case proto::schema::DataType::VarChar:
    return b == proto::schema::DataType::String ||
           b == proto::schema::DataType::VarChar ||
           b == proto::schema::DataType::JSON;
  case proto::schema::DataType::String:
    return b == proto::schema::DataType::String ||
           b == proto::schema::DataType::VarChar ||
           b == proto::schema::DataType::JSON;
  case proto::schema::DataType::JSON:
    return true;
  default:
    return false;
  }
}

bool canBeCompared(ExprWithDtype a, ExprWithDtype b) {
  if (a.dtype != proto::schema::DataType::Array &&
      b.dtype != proto::schema::DataType::Array) {
    return canBeComparedDataType(a.dtype, b.dtype);
  }

  if (a.dtype == proto::schema::DataType::Array &&
      b.dtype == proto::schema::DataType::Array) {
    return canBeComparedDataType(getArrayElementType(a.expr),
                                 getArrayElementType(b.expr));
  }

  if (a.dtype == proto::schema::DataType::Array) {
    return canBeComparedDataType(getArrayElementType(a.expr), b.dtype);
  }

  return canBeComparedDataType(b.dtype, getArrayElementType(b.expr));
}

ExprWithDtype toValueExpr(proto::plan::GenericValue *value) {
  auto expr = new proto::plan::Expr();
  auto value_expr = new proto::plan::ValueExpr();
  value_expr->set_allocated_value(value);

  if (value->has_bool_val()) {
    return ExprWithDtype(expr, proto::schema::DataType::Bool, false);
  }
  if (value->has_int64_val()) {
    return ExprWithDtype(expr, proto::schema::DataType::Int64, false);
  }
  if (value->has_float_val()) {
    return ExprWithDtype(expr, proto::schema::DataType::Float, false);
  }
  if (value->has_string_val()) {
    return ExprWithDtype(expr, proto::schema::DataType::String, false);
  }
  if (value->has_array_val()) {
    return ExprWithDtype(expr, proto::schema::DataType::Array, false);
  }
  assert(false);
}

proto::plan::Expr *HandleCompare(proto::plan::OpType op, ExprWithDtype a,
                                 ExprWithDtype b) {
  if (!a.expr->has_column_expr() || !b.expr->has_column_expr())
    assert(false);

  auto a_info = a.expr->column_expr().info();

  auto b_info = b.expr->column_expr().info();

  auto expr = new proto::plan::Expr();
  auto compare_expr = new proto::plan::CompareExpr();
  compare_expr->set_allocated_left_column_info(
      new proto::plan::ColumnInfo(a_info));
  compare_expr->set_allocated_right_column_info(
      new proto::plan::ColumnInfo(b_info));
  compare_expr->set_op(op);

  return expr;
}

proto::plan::GenericValue *castValue(proto::schema::DataType dtype,
                                     proto::plan::GenericValue *value) {
  if (dtype == proto::schema::DataType::JSON)
    return value;
  if (dtype == proto::schema::DataType::Array && value->has_array_val())
    return value;
  if (dtype == proto::schema::DataType::String && value->has_string_val())
    return value;
  if (dtype == proto::schema::DataType::Bool && value->has_bool_val())
    return value;
  if (dtype == proto::schema::DataType::Float ||
      dtype == proto::schema::DataType::Double) {
    if (value->has_float_val())
      return value;
    if (value->has_int64_val()) {
      auto value_ = new proto::plan::GenericValue();
      value_->set_float_val(double(value->int64_val()));
      return value_;
    }
  }

  if (dtype == proto::schema::DataType::Int8 ||
      dtype == proto::schema::DataType::Int16 ||
      dtype == proto::schema::DataType::Int32 ||
      dtype == proto::schema::DataType::Int64) {
    if (value->has_int64_val())
      return value;
  }

  assert(false);
}

proto::plan::Expr *
combineArrayLengthExpr(proto::plan::OpType op,
                       proto::plan::ArithOpType arith_op,
                       const proto::plan::ColumnInfo &info,
                       const proto::plan::GenericValue &value) {

  auto expr = new proto::plan::Expr();
  auto range_expr = new proto::plan::BinaryArithOpEvalRangeExpr();
  expr->set_allocated_binary_arith_op_eval_range_expr(range_expr);
  range_expr->set_op(op);
  range_expr->set_arith_op(arith_op);
  range_expr->set_allocated_value(new proto::plan::GenericValue(value));
  range_expr->set_allocated_column_info(new proto::plan::ColumnInfo(info));
  return expr;
}

proto::plan::Expr *
combineBinaryArithExpr(proto::plan::OpType op,
                       proto::plan::ArithOpType arith_op,
                       const proto::plan::ColumnInfo &info,
                       const proto::plan::GenericValue &operand,
                       const proto::plan::GenericValue &value) {
  auto data_type = info.data_type();
  if (data_type != proto::schema::DataType::Array &&
      info.nested_path_size() != 0) {
    data_type = info.element_type();
  }
  auto casted_value =
      castValue(data_type, new proto::plan::GenericValue(operand));
  auto expr = new proto::plan::Expr();
  auto range_expr = new proto::plan::BinaryArithOpEvalRangeExpr();
  expr->set_allocated_binary_arith_op_eval_range_expr(range_expr);
  range_expr->set_allocated_column_info(new proto::plan::ColumnInfo(info));
  range_expr->set_arith_op(arith_op);
  range_expr->set_allocated_right_operand(casted_value);
  range_expr->set_allocated_value(new proto::plan::GenericValue(value));
  range_expr->set_op(op);

  return expr;
}

proto::plan::Expr *
handleBinaryArithExpr(proto::plan::OpType op,
                      proto::plan::BinaryArithExpr *arith_expr,
                      proto::plan::ValueExpr *value_expr) {

  switch (op) {
  case proto::plan::OpType::Equal:
    break;
  case proto::plan::OpType::NotEqual:
    break;
  default:
    assert(false);
  }

  auto left_expr = arith_expr->left().column_expr();
  auto left_value = arith_expr->left().value_expr();
  auto right_expr = arith_expr->right().column_expr();
  auto right_value = arith_expr->right().value_expr();

  auto arith_op = arith_expr->op();

  if (arith_op == proto::plan::ArithOpType::ArrayLength) {
    return combineArrayLengthExpr(op, arith_op, left_expr.info(),
                                  value_expr->value());
  }
  if (arith_expr->left().has_column_expr() &&
      arith_expr->right().has_column_expr()) {
    assert(false);
  }
  if (arith_expr->left().has_value_expr() &&
      arith_expr->right().has_value_expr()) {
    assert(false);
  }
  if (arith_expr->left().has_column_expr() &&
      arith_expr->right().has_value_expr()) {
    return combineBinaryArithExpr(op, arith_op, left_expr.info(),
                                  right_value.value(), value_expr->value());
  }
  if (arith_expr->right().has_column_expr() &&
      arith_expr->left().has_value_expr()) {

    switch (arith_expr->op()) {
    case proto::plan::ArithOpType::Add:
      return combineBinaryArithExpr(op, arith_op, right_expr.info(),
                                    left_value.value(), value_expr->value());
    case proto::plan::ArithOpType::Mul:
      return combineBinaryArithExpr(op, arith_op, right_expr.info(),
                                    left_value.value(), value_expr->value());
    default:
      assert(false);
    }
  }
  assert(false);
}

proto::plan::Expr *handleCompareRightValue(proto::plan::OpType op,
                                           ExprWithDtype a, ExprWithDtype b) {
  auto data_type = a.dtype;
  if (data_type == proto::schema::DataType::Array &&
      a.expr->column_expr().info().nested_path_size() != 0)

  {
    data_type = a.expr->column_expr().info().element_type();
  }
  auto value = b.expr->value_expr().value();
  auto castedvalue = castValue(data_type, &value);
  if (a.expr->has_binary_expr()) {
    auto value_expr = new proto::plan::ValueExpr();
    value_expr->set_allocated_value(castedvalue);
    return handleBinaryArithExpr(
        op, new proto::plan::BinaryArithExpr(a.expr->binary_arith_expr()),
        value_expr);
  }

  assert(a.expr->has_column_expr());
  auto info = a.expr->column_expr().info();

  auto expr = new proto::plan::Expr();
  auto unary_range_expr = new proto::plan::UnaryRangeExpr();

  unary_range_expr->set_op(op);
  unary_range_expr->set_allocated_column_info(
      new proto::plan::ColumnInfo(info));

  unary_range_expr->set_allocated_value(castedvalue);

  return expr;
}

bool checkDirectComparisonBinaryField(proto::plan::ColumnInfo *info) {
  if (info->data_type() == proto::schema::DataType::Array &&
      info->nested_path_size() == 0) {
    return false;
  }
  return true;
}

} // namespace milvus
